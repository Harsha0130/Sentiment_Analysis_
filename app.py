from flask import Flask, render_template, redirect, request, url_for, session, flash
import nltk
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os, pickle, pandas as pd, base64, io, math, re, time
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service as ChromeService
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend suitable for Flask
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud # For generating word clouds
from collections import Counter # For n-gram analysis
from nltk.corpus import stopwords # For filtering common words
from webdriver_manager.chrome import ChromeDriverManager


# Download NLTK resources if not already present
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    nltk.download('vader_lexicon')
try:
    nltk.data.find('corpora/wordnet.zip')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/stopwords.zip')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# -------------------------------------------
# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# -------------------------------------------
# Model Load
# Ensure 'model' directory exists before trying to load the model
if not os.path.exists('model') or not os.path.exists('model/model.pkl'):
    print("WARNING: 'model/model.pkl' not found. Please run train_model.py first.")
    # Create dummy model/vectorizer to prevent immediate crash, but functionality will be limited
    model = None
    tfidf_vectorizer = None
else:
    with open('model/model.pkl', 'rb') as f:
        model, tfidf_vectorizer = pickle.load(f)

sid = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
english_stopwords = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", str(text)).lower()
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in english_stopwords])
    return text

def predict_sentiment(text):
    if model is None or tfidf_vectorizer is None:
        print("ERROR: Model or TF-IDF vectorizer not loaded. Cannot predict sentiment.")
        return "Unknown" # Or handle as appropriate

    cleaned = preprocess_text(text)
    vector = tfidf_vectorizer.transform([cleaned])
    return model.predict(vector)[0]

def generate_charts(sentiment_counts):
    total = sum(sentiment_counts.values())
    if total == 0:
        return "", ""

    sentiment_counts = {k: v for k, v in sentiment_counts.items() if not (isinstance(v, float) and math.isnan(v))}

    # Bar Chart
    bar_fig = plt.figure(figsize=(5, 3))
    plt.bar(sentiment_counts.keys(), sentiment_counts.values(), color=['green', 'red'])
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    bar_img = io.BytesIO()
    bar_fig.savefig(bar_img, format='png')
    bar_img.seek(0)
    bar_chart = base64.b64encode(bar_img.getvalue()).decode('utf-8')
    plt.close(bar_fig)

    # Pie Chart
    pie_fig = plt.figure(figsize=(5, 5))
    plt.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct='%1.1f%%', colors=['green', 'red'])
    plt.title('Sentiment Distribution')
    pie_img = io.BytesIO()
    pie_fig.savefig(pie_img, format='png')
    pie_img.seek(0)
    pie_chart = base64.b64encode(pie_img.getvalue()).decode('utf-8')
    plt.close(pie_fig)

    return bar_chart, pie_chart

def generate_word_cloud(text_data, title="Word Cloud"):
    if not text_data:
        return ""
    
    # Ensure text_data is a single string
    combined_text = " ".join(text_data)
    
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          stopwords=english_stopwords, min_font_size=10).generate(combined_text)
    
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    plt.close()
    return img_base64

def generate_ngram_chart(text_data, n=2, title="Top N-grams"):
    if not text_data:
        return ""

    # Combine all text into a single string
    combined_text = " ".join(text_data)
    
    # Tokenize and filter stopwords
    words = [word for word in re.findall(r'\b\w+\b', combined_text.lower()) if word not in english_stopwords]
    
    # Generate n-grams
    ngrams = Counter(tuple(words[i:i+n]) for i in range(len(words)-n+1))
    
    # Get top 10 n-grams
    top_ngrams = ngrams.most_common(10)
    
    if not top_ngrams:
        return ""

    labels = [" ".join(ngram[0]) for ngram in top_ngrams]
    counts = [ngram[1] for ngram in top_ngrams]

    plt.figure(figsize=(10, 6))
    plt.barh(labels, counts, color='skyblue')
    plt.xlabel('Frequency')
    plt.ylabel('N-gram')
    plt.title(title)
    plt.gca().invert_yaxis() # Puts the highest bar at the top
    plt.tight_layout()

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    plt.close()
    return img_base64

def generate_rating_distribution_chart(df):
    if 'Reviewer_Rating' not in df.columns:
        return ""

    # Convert ratings to numeric, coercing errors to NaN
    df['Reviewer_Rating_Numeric'] = pd.to_numeric(df['Reviewer_Rating'], errors='coerce')
    
    # Filter out NaN values and count occurrences for each star rating (1-5)
    rating_counts = df['Reviewer_Rating_Numeric'].dropna().astype(int).value_counts().sort_index()

    # Ensure all ratings from 1 to 5 are present, fill missing with 0
    full_rating_counts = {i: rating_counts.get(i, 0) for i in range(1, 6)}
    
    if sum(full_rating_counts.values()) == 0:
        return ""

    labels = [f"{i} Star" for i in range(1, 6)]
    counts = [full_rating_counts[i] for i in range(1, 6)]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, counts, color=['#FF4500', '#FF8C00', '#FFD700', '#ADFF2F', '#32CD32']) # Colors for 1-5 stars
    plt.title('Rating Distribution')
    plt.xlabel('Star Rating')
    plt.ylabel('Number of Reviews')
    plt.tight_layout()

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    plt.close()
    return img_base64

def generate_product_conclusion(df, sentiment_counts):
    total_reviews = len(df)
    if total_reviews == 0:
        return "No reviews available for analysis to draw a conclusion."

    # Calculate average rating
    df['Reviewer_Rating_Numeric'] = pd.to_numeric(df['Reviewer_Rating'], errors='coerce')
    average_rating = df['Reviewer_Rating_Numeric'].mean()

    # Calculate sentiment percentages
    positive_percent = (sentiment_counts.get('Positive', 0) / total_reviews) * 100
    negative_percent = (sentiment_counts.get('Negative', 0) / total_reviews) * 100

    conclusion = f"Based on {total_reviews} reviews, the product generally has a "
    
    if average_rating >= 4.0:
        conclusion += f"strong positive reception with an average rating of {average_rating:.2f} stars. "
    elif average_rating >= 3.0:
        conclusion += f"mixed to positive reception with an average rating of {average_rating:.2f} stars. "
    else:
        conclusion += f"tendency towards negative reception with an average rating of {average_rating:.2f} stars. "

    conclusion += f"Approximately {positive_percent:.1f}% of reviews are positive and {negative_percent:.1f}% are negative. "

    if positive_percent > negative_percent * 2:
        conclusion += "Customers are highly satisfied."
    elif negative_percent > positive_percent * 2:
        conclusion += "There are significant areas for improvement based on negative feedback."
    elif positive_percent > negative_percent:
        conclusion += "Overall sentiment is positive, but some negative points exist."
    else:
        conclusion += "Sentiment is relatively balanced, indicating diverse opinions."

    return conclusion


# -------------------------------------------
# Scrape Flipkart Reviews
def scrape_flipkart_reviews(product_url, output_filename='data/flipkart_reviews.csv', max_pages=10): # Increased max_pages
    options = Options()
    options.add_argument('--headless') # Enabled headless mode for faster execution
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--start-maximized') 
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage') 
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
    options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
    options.add_argument("blink-settings=imagesEnabled=false")
    options.add_argument("disable-features=IsolateOrigins,site-per-process")

    # Use ChromeDriverManager to automatically manage chromedriver
    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    reviews = []
    
    try:
        # --- Step 1: Extract PID and other necessary parts from the product URL and construct reviews URL ---
        print(f"Original product URL: {product_url}")
        
        match = re.search(r'flipkart\.com/([^/]+)/p/(itm[a-zA-Z0-9]+)\?pid=([^&]+)', product_url)
        
        if not match:
            print("ERROR: Could not parse product URL. Please provide a valid Flipkart product URL in the format: https://www.flipkart.com/{product-name}/p/{product-id}?pid={pid}&...")
            driver.quit()
            return output_filename

        product_name_slug = match.group(1)
        product_id_from_url = match.group(2)
        pid = match.group(3)

        reviews_url = f"https://www.flipkart.com/{product_name_slug}/product-reviews/{product_id_from_url}?pid={pid}&marketplace=FLIPKART"
        
        print(f"Navigating directly to reviews URL: {reviews_url}")
        driver.get(reviews_url)
        time.sleep(2) # Reduced initial sleep for the reviews page to load fully

        # --- Step 2: Wait for the review cards to appear on the reviews page ---
        print("Waiting for review cards to appear on the reviews page (max 15 seconds)...") # Reduced wait time
        try:
            WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.XPATH, "//div[@class='col EPCmJX Ma1fCG']")))
            print("Successfully found at least one review card on the reviews page.")
            time.sleep(1) # Small buffer after finding initial card
        except Exception as e:
            print(f"⚠️ No review cards found on the reviews page within timeout: {e}")
            print("This might mean the reviews page structure has changed or reviews are not available. Exiting scraper.")
            driver.quit()
            return output_filename 


        # --- Step 3: Loop through pages and scrape reviews ---
        for page in range(1, max_pages + 1):
            print(f"➡ Scraping page {page}")

            # Scroll down to load more reviews
            last_height = driver.execute_script("return document.body.scrollHeight")
            scroll_attempts = 0
            while True:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1) # Further reduced sleep during scrolling
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    scroll_attempts += 1
                    if scroll_attempts > 2: # Try scrolling a few times before giving up
                        break
                else:
                    scroll_attempts = 0 # Reset if new content loaded
                last_height = new_height
            print("Scrolled to bottom of the page.")
            time.sleep(0.5) # Additional small delay after scrolling

            try:
                # Ensure review cards are present after scrolling
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//div[@class='col EPCmJX Ma1fCG']"))) # Reduced wait time
                cards = driver.find_elements(By.XPATH, "//div[@class='col EPCmJX Ma1fCG']")
                print(f"Found {len(cards)} potential review cards on page {page}.")
            except Exception as e:
                print(f"⚠️ No review cards found on page {page} after scrolling within timeout: {e}. Ending scraping for this product.")
                break 

            if not cards:
                print(f"⚠️ No review cards found on page {page}. Ending scraping.")
                break 

            for card in cards:
                try:
                    name_elem = card.find_elements(By.CSS_SELECTOR, "p._2NsDsF.AwS1CA") 
                    name = name_elem[0].text.strip() if name_elem else "N/A"

                    rating_elem = card.find_elements(By.CSS_SELECTOR, "div.XQDdHH.Ga3i8K") 
                    rating = rating_elem[0].text.strip().replace("Review", "").strip() if rating_elem else "N/A"

                    title_elem = card.find_elements(By.CSS_SELECTOR, "p.z9E0IG") 
                    title = title_elem[0].text.strip() if title_elem else "N/A"

                    review_elem = card.find_elements(By.CSS_SELECTOR, "div.ZmyHeo > div > div") 
                    review = review_elem[0].text.strip() if review_elem else "N/A"
                    
                    place = "N/A"
                    date = "N/A"
                    
                    place_elem = card.find_elements(By.CSS_SELECTOR, "p.MztJPv span:nth-child(2)")
                    if place_elem:
                        place = place_elem[0].text.strip().replace(",", "")

                    date_elem = card.find_elements(By.XPATH, ".//p[@class='_2NsDsF' and not(contains(@class, 'AwS1CA'))]")
                    if date_elem:
                        date = date_elem[-1].text.strip()


                    upvotes = "0"
                    downvotes = "0"
                    try:
                        upvote_span = card.find_elements(By.XPATH, ".//div[@class='_6kK6mk']/span[@class='tl9VpF']")
                        if upvote_span:
                            upvotes = upvote_span[0].text.strip()
                    except Exception as e:
                        print(f"Error getting upvotes: {e}")
                        pass 
                    
                    try:
                        downvote_span = card.find_elements(By.XPATH, ".//div[@class='_6kK6mk aQymJL']/span[@class='tl9VpF']")
                        if downvote_span:
                            downvotes = downvote_span[0].text.strip()
                    except Exception as e:
                        print(f"Error getting downvotes: {e}")
                        pass 

                    reviews.append({
                        'Reviewer_Name': name,
                        'Reviewer_Rating': rating,
                        'Review_Title': title,
                        'Review_Text': review,
                        'Place_of_Review': place,
                        'Date_of_Review': date,
                        'Up_Votes': upvotes,
                        'Down_Votes': downvotes
                    })
                    print(f"✅ Saved review: {title[:30]}... by {name}")

                except Exception as e:
                    print(f"⚠️ Skipping a review card on page {page} due to error during extraction: {e}")
                    continue
            
            # --- Step 4: Navigate to the next page ---
            try:
                print(f"Attempting to find 'Next' button for page {page + 1}...")
                next_button = WebDriverWait(driver, 7).until( # Reduced wait time
                    EC.element_to_be_clickable((By.XPATH, "//a[@class='_9QVEpD' and span[text()='Next']] | //a[@class='_1LKTO3' and span[text()='Next']] | //a[contains(@class, '_1LKTO3')]//span[text()='Next'] | //a[text()='Next']"))
                )
                driver.execute_script("arguments[0].click();", next_button)
                print(f"Clicked 'Next' button. Waiting for page {page + 1} to load reviews...")
                time.sleep(3) # Reduced sleep for page navigation
            except Exception as e:
                print(f"❌ Could not find or click 'Next' button or no more pages for product {product_url}: {e}")
                break 

    except Exception as e:
        print(f"An unexpected error occurred during scraping process for {product_url}: {e}")
    finally:
        driver.quit() 

    # Save to CSV
    if reviews:
        df = pd.DataFrame(reviews)
        os.makedirs('data', exist_ok=True)
        df.to_csv(output_filename, index=False)
        print(f"✅ Scraped {len(reviews)} reviews. Saved to {output_filename}")
    else:
        print(f"⚠️ No reviews collected for {product_url}. Saving only headers to {output_filename}.")
        os.makedirs('data', exist_ok=True)
        pd.DataFrame(columns=[
            'Reviewer_Name', 'Reviewer_Rating', 'Review_Title', 'Review_Text',
            'Place_of_Review', 'Date_of_Review', 'Up_Votes', 'Down_Votes'
        ]).to_csv(output_filename, index=False)

    return output_filename

# -------------------------------------------
# Routes (rest of your app.py code remains the same)
@app.route('/')
def home():
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    return render_template('index.html', csv_files=csv_files)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        hashed_pw = generate_password_hash(password)

        if User.query.filter_by(email=email).first():
            flash("Email already registered", "warning")
            return redirect(url_for('signup'))

        new_user = User(email=email, password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()
        flash("Signup successful! Please log in.", "success")
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form['email']).first()
        if user and check_password_hash(user.password, request.form['password']):
            login_user(user)
            flash("Login successful!", "success")
            return redirect(url_for('home')) 
        else:
            flash("Invalid credentials", "danger")
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Logged out successfully", "info")
    return redirect(url_for('home'))

@app.route('/scrape', methods=['GET', 'POST'])
@login_required
def scrape_reviews():
    if request.method == 'POST':
        url = request.form['url']
        filename = scrape_flipkart_reviews(url)
        return redirect(url_for('analyze_data', filename=os.path.basename(filename)))
    return render_template('scrape.html')

@app.route('/analyze/<filename>')
@login_required
def analyze_data(filename):
    df = pd.read_csv(os.path.join('data', filename))

    if 'Review_Text' not in df.columns:
        flash("The CSV file does not contain a 'Review_Text' column, cannot analyze sentiment.", "danger")
        return redirect(url_for('home')) 

    if df.empty:
        flash("No reviews found in the scraped data. Please try scraping again or check the product URL.", "warning")
        # Ensure all chart variables are passed, even if empty, to prevent template errors
        return render_template('analysis_result.html', results=[], filename=filename, 
                               bar_chart="", pie_chart="", rating_distribution_chart="",
                               positive_word_cloud="", negative_word_cloud="", 
                               bigram_chart="", trigram_chart="", product_conclusion="No data to analyze.")


    results = []
    sentiment_counts = {'Positive': 0, 'Negative': 0}
    positive_reviews_text = []
    negative_reviews_text = []

    for _, row in df.iterrows():
        review_text = str(row.get('Review_Text', '')) 
        
        if review_text.strip() and model is not None and tfidf_vectorizer is not None:
            sentiment = predict_sentiment(review_text)
            row_dict = row.to_dict()
            row_dict['sentiment'] = sentiment
            results.append(row_dict)
            sentiment_counts[sentiment] += 1
            if sentiment == 'Positive':
                positive_reviews_text.append(review_text)
            elif sentiment == 'Negative':
                negative_reviews_text.append(review_text)
        else:
            row_dict = row.to_dict()
            row_dict['sentiment'] = "Unknown"
            results.append(row_dict)


    bar_chart, pie_chart = generate_charts(sentiment_counts)
    rating_distribution_chart = generate_rating_distribution_chart(df)
    positive_word_cloud = generate_word_cloud(positive_reviews_text, title="Positive Reviews Word Cloud")
    negative_word_cloud = generate_word_cloud(negative_reviews_text, title="Negative Reviews Word Cloud")
    
    # N-gram analysis
    all_reviews_text = df['Review_Text'].dropna().tolist()
    bigram_chart = generate_ngram_chart(all_reviews_text, n=2, title="Top 10 Bigrams")
    trigram_chart = generate_ngram_chart(all_reviews_text, n=3, title="Top 10 Trigrams")

    # Generate Product Conclusion
    product_conclusion = generate_product_conclusion(df, sentiment_counts)


    return render_template('analysis_result.html',
                           results=results,
                           filename=filename,
                           bar_chart=bar_chart,
                           pie_chart=pie_chart,
                           rating_distribution_chart=rating_distribution_chart,
                           positive_word_cloud=positive_word_cloud,
                           negative_word_cloud=negative_word_cloud,
                           bigram_chart=bigram_chart,
                           trigram_chart=trigram_chart,
                           product_conclusion=product_conclusion)

# -------------------------------------------
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)