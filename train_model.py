import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os # Import os to ensure directory exists

# Load the data
# Ensure these CSV files exist in your 'data' folder or provide the correct path
tawa_df = pd.read_csv("data/reviews_tawa.csv")
tea_df = pd.read_csv("data/reviews_tea.csv")

# Data Preprocessing
combined_df = pd.concat([tawa_df, tea_df])

# Text Cleaning
def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", str(text))
    text = text.lower()
    return text

# Text Normalization
def normalize_text(text):
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    normalized_words = [lemmatizer.lemmatize(word) for word in words]
    normalized_text = " ".join(normalized_words)
    return normalized_text

# Combine Clean & Normalization
def preprocess_text(text):
    cleaned_text = clean_text(text)
    normalized_text = normalize_text(cleaned_text)
    return normalized_text

# Ensure 'Review_Text' is string and handle NaNs before preprocessing and sentiment analysis
# This line is crucial for preventing the 'float' object has no attribute 'encode' error
combined_df['Review_Text'] = combined_df['Review_Text'].astype(str).fillna('')

combined_df['processed_text'] = combined_df['Review_Text'].apply(preprocess_text)

# Sentiment Analysis
sid = SentimentIntensityAnalyzer()

# Calculate sentiment score using VADER
combined_df['sentiment_score'] = combined_df['Review_Text'].apply(lambda x: sid.polarity_scores(x)['compound'])

# Add sentiment label (Positive if score > 0, Negative otherwise)
combined_df['sentiment'] = combined_df['sentiment_score'].apply(lambda score: 'Positive' if score > 0 else 'Negative')

# Check if sentiment column is created successfully
# Corrected column names in print statement for consistency
print(combined_df[['Review_Text', 'sentiment_score', 'sentiment']].head())

# Split the data for training
X = combined_df['processed_text']
y = combined_df['sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Test the model
X_test_tfidf = tfidf_vectorizer.transform(X_test)
y_pred = model.predict(X_test_tfidf)

# Classification report
print("Classification Report for Logistic Regression:")
print(classification_report(y_test, y_pred))

# Create 'model' directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Save the trained model and vectorizer to disk
with open('model/model.pkl', 'wb') as f:
    pickle.dump((model, tfidf_vectorizer), f)

print("\nModel and TF-IDF vectorizer saved to 'model/model.pkl'")