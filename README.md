# South African News Sentiment Analyzer

This is an **NLP-based sentiment analysis** tool designed to classify South African news headlines as either **positive** or **negative**. The project aims to leverage text processing and machine learning models to predict sentiment, providing insights into public opinion based on news articles.

## Project Overview

The goal of this project is to analyze a small dataset of South African news headlines and predict the sentiment of the text. The sentiment can be either **positive** or **negative**. This was achieved by preprocessing the text, applying a TF-IDF vectorizer, and using a logistic regression model.

### Technologies Used:
- **Python Libraries:** 
  - Pandas
  - NLTK
  - Scikit-learn
  - Streamlit
- **Machine Learning Models:** Logistic Regression, Random Forest Classifier
- **Text Preprocessing:** Tokenization, Stopword removal, TF-IDF vectorization

## Dataset

The dataset used for training and testing contains **50** South African news headlines. The labels are binary: `positive` or `negative`.

## Model Performance

The model was evaluated on a test set with the following metrics:

| Metric      | Negative | Positive | Accuracy |
|-------------|----------|----------|----------|
| **Precision**  | 0.25     | 0.00     | 0.25     |
| **Recall**     | 1.00     | 0.00     | 0.25     |
| **F1-score**   | 0.40     | 0.00     | 0.10     |

**Note:** The model’s performance is quite low due to the small dataset size. The next step is to expand the dataset and experiment with more sophisticated models.

## Installation

To run this project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sentiment-analyzer.git
