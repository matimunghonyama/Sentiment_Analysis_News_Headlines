import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

class SentimentAnalyzer:
    def __init__(self):
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('stopwords')
        
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.model = LogisticRegression()
    
    def preprocess_text(self, text):
        """Preprocess text for analysis"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        return ' '.join(tokens)
    
    def train(self, df):
        """Train the sentiment analysis model"""
        # Preprocess headlines
        processed_headlines = df['headline'].apply(self.preprocess_text)
        
        # Create features
        X = self.vectorizer.fit_transform(processed_headlines)
        y = df['sentiment']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        print("\nModel Performance:")
        print(classification_report(y_test, y_pred))
    
    def predict_sentiment(self, headline):
        """Predict sentiment for a given headline"""
        processed_headline = self.preprocess_text(headline)
        X = self.vectorizer.transform([processed_headline])
        return self.model.predict(X)[0]

def main():
    # Load dataset
    df = pd.read_csv('basic_sentiment_dataset.csv')
    
    # Initialize and train analyzer
    analyzer = SentimentAnalyzer()
    analyzer.train(df)
    
    # Test prediction
    test_headline = "South Africa's economy shows signs of recovery"
    prediction = analyzer.predict_sentiment(test_headline)
    print(f"\nTest prediction for: {test_headline}")
    print(f"Predicted sentiment: {prediction}")

if __name__ == "__main__":
    main()