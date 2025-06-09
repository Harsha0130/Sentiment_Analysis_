# Sentiment Analysis Project

A comprehensive Flask web application for sentiment analysis of Flipkart product reviews with web scraping capabilities.

## Features

- **User Authentication**: Secure login/signup system
- **Web Scraping**: Automated Flipkart review scraping using Selenium
- **Sentiment Analysis**: Machine learning-based sentiment classification
- **Data Visualization**: Interactive charts and word clouds
- **Responsive Design**: Modern, mobile-friendly interface

## Project Structure

```
sentiment-analysis/
├── app.py                 # Main Flask application
├── train_model.py         # Model training script
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── templates/            # HTML templates
│   ├── index.html
│   ├── login.html
│   ├── signup.html
│   ├── scrape.html
│   └── analysis_result.html
├── data/                 # CSV data files (created automatically)
├── model/                # Trained models (created automatically)
└── static/               # Static files (CSS, JS, images)
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd sentiment-analysis
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (optional - will use sample data if no CSV files exist):
   ```bash
   python train_model.py
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

5. **Access the application**:
   Open your browser and go to `http://localhost:5000`

## Usage

### 1. User Registration/Login
- Create an account or login with existing credentials
- Authentication is required for scraping and analysis features

### 2. Scraping Reviews
- Navigate to the "Scrape Flipkart Reviews" section
- Enter a valid Flipkart product URL
- The system will automatically scrape reviews and save them as CSV

### 3. Analyzing Data
- View existing CSV files on the dashboard
- Click "Analyze" to perform sentiment analysis
- View comprehensive results including:
  - Sentiment distribution charts
  - Word clouds for positive/negative reviews
  - N-gram analysis
  - Detailed review table with sentiment scores

## Technical Details

### Machine Learning Model
- **Algorithm**: Logistic Regression
- **Features**: TF-IDF vectorization
- **Preprocessing**: Text cleaning, lemmatization, stopword removal
- **Sentiment Labels**: Generated using VADER sentiment analyzer

### Web Scraping
- **Tool**: Selenium WebDriver with Chrome
- **Target**: Flipkart product review pages
- **Data Extracted**: 
  - Reviewer name and rating
  - Review title and text
  - Location and date
  - Upvotes/downvotes

### Visualizations
- **Charts**: Bar charts, pie charts, rating distributions
- **Word Clouds**: Separate clouds for positive and negative reviews
- **N-gram Analysis**: Top bigrams and trigrams

## Configuration

### Environment Variables
- `SECRET_KEY`: Flask secret key for sessions
- `SQLALCHEMY_DATABASE_URI`: Database connection string

### Model Training
- Modify `train_model.py` to use your own training data
- Place CSV files with 'Review_Text' column in the `data/` directory
- Run the training script to generate a new model

## Dependencies

- **Flask**: Web framework
- **SQLAlchemy**: Database ORM
- **Selenium**: Web scraping
- **scikit-learn**: Machine learning
- **NLTK**: Natural language processing
- **Matplotlib**: Data visualization
- **WordCloud**: Word cloud generation
- **Pandas**: Data manipulation

## Browser Requirements

- Chrome browser (for Selenium WebDriver)
- JavaScript enabled
- Modern browser with CSS3 support

## Troubleshooting

### Common Issues

1. **ChromeDriver Issues**:
   - The app uses `webdriver-manager` to automatically download ChromeDriver
   - Ensure Chrome browser is installed

2. **Model Not Found**:
   - Run `python train_model.py` to create the model
   - Check that `model/model.pkl` exists

3. **Scraping Failures**:
   - Verify the Flipkart URL format
   - Check internet connection
   - Some products may have different page structures

### Performance Tips

- Use headless Chrome for faster scraping
- Limit the number of pages scraped for large datasets
- Consider using a more powerful machine for model training

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational purposes. Please respect website terms of service when scraping data.

## Future Enhancements

- Support for other e-commerce platforms
- Real-time sentiment analysis
- Advanced ML models (BERT, transformers)
- API endpoints for external integration
- Docker containerization
- Cloud deployment options
