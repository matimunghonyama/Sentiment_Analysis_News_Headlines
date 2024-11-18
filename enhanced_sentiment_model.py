import pandas as pd
import numpy as np
import streamlit as st
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

class ImprovedSentimentAnalyzer:
    def __init__(self):
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.stop_words = set(stopwords.words('english'))

    def advanced_preprocess(self, text):
        # More robust preprocessing
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        
        # Enhanced token filtering
        tokens = [
            token for token in tokens 
            if token not in self.stop_words and len(token) > 1
        ]
        
        return ' '.join(tokens)

    def create_pipeline(self, df):
        # Compute class weights to handle imbalance
        class_weights = compute_class_weight(
            class_weight='balanced', 
            classes=np.unique(df['sentiment']), 
            y=df['sentiment']
        )
        class_weight_dict = dict(zip(np.unique(df['sentiment']), class_weights))

        # Create pipeline with grid search
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 2),  # Include bigrams
                max_features=2000
            )),
            ('classifier', LogisticRegression(
                class_weight=class_weight_dict,
                max_iter=2000
            ))
        ])

        # Hyperparameter grid
        param_grid = {
            'classifier__C': [0.1, 1, 10],
            'classifier__solver': ['liblinear', 'lbfgs']
        }

        # Grid search with cross-validation
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=5, 
            scoring='f1_weighted'
        )

        return grid_search

    def train_and_evaluate(self, df):
        # Preprocess headlines
        processed_headlines = df['headline'].apply(self.advanced_preprocess)
        X = processed_headlines
        y = df['sentiment']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Create and fit the pipeline
        grid_search = self.create_pipeline(df)
        grid_search.fit(X_train, y_train)

        # Best model predictions
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        # Detailed evaluation
        st.text("\nImproved Model Performance:")
        st.text(classification_report(y_test, y_pred))
        st.text("\nConfusion Matrix:")
        st.text(confusion_matrix(y_test, y_pred))
        st.text("\nBest Hyperparameters:")
        st.text(grid_search.best_params_)

        return best_model

def main():
    st.title("🌍 Enhanced South African News Sentiment Analyzer 📰")
    
    @st.cache_data
    def load_data():
        return pd.read_csv('basic_sentiment_dataset.csv')
    
    df = load_data()
    analyzer = ImprovedSentimentAnalyzer()
    
    st.write("Training the improved model...")
    best_model = analyzer.train_and_evaluate(df)
    st.success("Enhanced model training complete!")

if __name__ == "__main__":
    main()