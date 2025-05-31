from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import pandas as pd
from webdriver_manager.chrome import ChromeDriverManager

def scrape_flipkart_reviews(product_url, output_filename='data/flipkart_reviews.csv', max_pages=2):
    options = Options()
    # Comment out headless mode to see the browser
    # options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)  # Use WebDriverManager for automatic driver installation
    driver.get(product_url)
    time.sleep(3)

    try:
        print("Trying to click 'View all reviews'...")
        view_all = driver.find_element(By.XPATH, "//div[contains(text(),'View all reviews')]")
        driver.execute_script("arguments[0].click();", view_all)
        time.sleep(3)
    except Exception as e:
        print("❌ Could not find 'View all reviews' button:", e)

    reviews = []

    for page in range(max_pages):
        print(f"➡ Scraping page {page + 1}")
        review_cards = driver.find_elements(By.CSS_SELECTOR, "div._1AtVbE")

        print("Found", len(review_cards), "review containers")

        for card in review_cards:
            try:
                rating = card.find_element(By.CSS_SELECTOR, "div._3LWZlK").text
                title = card.find_element(By.CSS_SELECTOR, "p._2-N8zT").text
                review = card.find_element(By.CSS_SELECTOR, "div.t-ZTKy div").text
                name = card.find_element(By.CSS_SELECTOR, "p._2sc7ZR._2V5EHH").text
                location_date = card.find_element(By.CSS_SELECTOR, "p._2mcZGG").text
                upvotes = card.find_element(By.XPATH, ".//span[contains(text(),'Helpful')]/preceding-sibling::span").text
                downvotes = card.find_element(By.XPATH, ".//span[contains(text(),'Not Helpful')]/preceding-sibling::span").text

                # Basic location/date parsing
                place = location_date.split(',')[0].replace('Certified Buyer', '').strip()
                date = location_date.split(',')[-1].strip()

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

                print(f"✅ Review by {name}: {title}")

            except Exception as e:
                print("Skipping review card:", e)
                continue

        # Try clicking next
        try:
            next_btn = driver.find_elements(By.CSS_SELECTOR, "a._1LKTO3")[-1]
            if 'Next' in next_btn.text:
                driver.execute_script("arguments[0].click();", next_btn)
                time.sleep(2)
            else:
                break
        except Exception as e:
            print("❌ Could not click next:", e)
            break

    driver.quit()

    if reviews:
        df = pd.DataFrame(reviews)
        df.to_csv(output_filename, index=False)
        print(f"✅ Scraped {len(reviews)} reviews. Saved to {output_filename}")
    else:
        print("⚠️ No reviews were scraped.")

# RUN THIS:
scrape_flipkart_reviews("https://www.flipkart.com/nothing-phone-3a/p/itm8150b2c810f5b?pid=MOBH8G3P6UXPEFSZ")

