import pandas as pd
import streamlit as st
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Define the SentimentAnalyzer class
class SentimentAnalyzer:
    def __init__(self):
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.model = LogisticRegression(max_iter=1000)

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        return ' '.join(tokens)

    def train(self, df):
        processed_headlines = df['headline'].apply(self.preprocess_text)
        X = self.vectorizer.fit_transform(processed_headlines)
        y = df['sentiment']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        st.text("\nModel Performance:")
        st.text(classification_report(y_test, y_pred))

    def predict_sentiment(self, headline):
        processed_headline = self.preprocess_text(headline)
        X = self.vectorizer.transform([processed_headline])
        return self.model.predict(X)[0]

# Define the AspectAnalyzer class
class AspectAnalyzer:
    def analyze_aspects(self, headline):
        return {
            'headline': headline,
            'government': self._analyze_government(headline),
            'economy': self._analyze_economy(headline),
            'infrastructure': self._analyze_infrastructure(headline)
        }

    def _analyze_government(self, headline):
        positive_keywords = ['launches', 'improves', 'develops', 'supports', 'initiative']
        negative_keywords = ['fails', 'corrupt', 'ineffective', 'challenges']
        if any(word in headline for word in ['government', 'legislation', 'policy']):
            if any(word in headline for word in positive_keywords):
                return 'positive'
            if any(word in headline for word in negative_keywords):
                return 'negative'
        return 'neutral'

    def _analyze_economy(self, headline):
        positive_keywords = ['growth', 'recovery', 'exceeds', 'increases', 'milestone']
        negative_keywords = ['decline', 'recession', 'struggles', 'loss', 'downturn']
        if any(word in headline for word in ['economy', 'business', 'jobs', 'market']):
            if any(word in headline for word in positive_keywords):
                return 'positive'
            if any(word in headline for word in negative_keywords):
                return 'negative'
        return 'neutral'

    def _analyze_infrastructure(self, headline):
        positive_keywords = ['improves', 'develops', 'upgrades', 'expansion', 'new']
        negative_keywords = ['disrupts', 'delays', 'restrictions', 'fails', 'challenges']
        if any(word in headline for word in ['infrastructure', 'transport', 'water', 'electricity']):
            if any(word in headline for word in positive_keywords):
                return 'positive'
            if any(word in headline for word in negative_keywords):
                return 'negative'
        return 'neutral'

# Streamlit app
def main():
    st.title("🌍 South African News Sentiment Chatbot 📰")
    st.write("Analyze the sentiment and aspects of news headlines.")

    # Load dataset
    @st.cache
    def load_data():
        return pd.read_csv('basic_sentiment_dataset.csv')
    
    df = load_data()

    # Initialize analyzers
    sentiment_analyzer = SentimentAnalyzer()
    aspect_analyzer = AspectAnalyzer()

    # Train the sentiment model
    st.write("Training the model...")
    sentiment_analyzer.train(df)
    st.success("Model training complete!")

    # User input
    headline = st.text_input("📝 Enter a news headline:")
    
    if headline:
        # Analyze headline
        overall_sentiment = sentiment_analyzer.predict_sentiment(headline)
        aspect_results = aspect_analyzer.analyze_aspects(headline)

        # Display results
        st.subheader("📊 Analysis Results")
        st.write(f"**Headline:** {headline}")
        st.write(f"**Overall Sentiment:** {overall_sentiment.capitalize()}")

        st.subheader("Aspect-Based Insights:")
        for aspect, sentiment in aspect_results.items():
            st.write(f"- **{aspect.capitalize()}**: {sentiment.capitalize()}")

if __name__ == "__main__":
    main()
