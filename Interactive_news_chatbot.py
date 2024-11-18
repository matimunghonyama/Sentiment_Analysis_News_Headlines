import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import random

class SentimentAnalyzer:
    def __init__(self):
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.model = LogisticRegression(max_iter=1000)
    
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

class AspectAnalyzer:
    def analyze_aspects(self, headline):
        """Analyze sentiment for each aspect of a headline"""
        return {
            'headline': headline,
            'government': self._analyze_government(headline),
            'economy': self._analyze_economy(headline),
            'infrastructure': self._analyze_infrastructure(headline)
        }
    
    def _analyze_government(self, headline):
        """Analyze government-related sentiment"""
        headline = headline.lower()
        positive_keywords = ['launches', 'improves', 'develops', 'supports', 'initiative']
        negative_keywords = ['fails', 'corrupt', 'ineffective', 'challenges']
        
        if any(word in headline for word in ['government', 'legislation', 'policy']):
            if any(word in headline for word in positive_keywords):
                return 'positive'
            if any(word in headline for word in negative_keywords):
                return 'negative'
        return 'neutral'
    
    def _analyze_economy(self, headline):
        """Analyze economy-related sentiment"""
        headline = headline.lower()
        positive_keywords = ['growth', 'recovery', 'exceeds', 'increases', 'milestone']
        negative_keywords = ['decline', 'recession', 'struggles', 'loss', 'downturn']
        
        if any(word in headline for word in ['economy', 'business', 'jobs', 'market']):
            if any(word in headline for word in positive_keywords):
                return 'positive'
            if any(word in headline for word in negative_keywords):
                return 'negative'
        return 'neutral'
    
    def _analyze_infrastructure(self, headline):
        """Analyze infrastructure-related sentiment"""
        headline = headline.lower()
        positive_keywords = ['improves', 'develops', 'upgrades', 'expansion', 'new']
        negative_keywords = ['disrupts', 'delays', 'restrictions', 'fails', 'challenges']
        
        if any(word in headline for word in ['infrastructure', 'transport', 'water', 'electricity']):
            if any(word in headline for word in positive_keywords):
                return 'positive'
            if any(word in headline for word in negative_keywords):
                return 'negative'
        return 'neutral'

class NewsSentimentChatbot:
    def __init__(self, dataset_path='basic_sentiment_dataset.csv'):
        # Load dataset
        self.df = pd.read_csv(dataset_path)
        
        # Initialize analyzers
        self.sentiment_analyzer = SentimentAnalyzer()
        self.aspect_analyzer = AspectAnalyzer()
        
        # Train sentiment model
        self.sentiment_analyzer.train(self.df)
        
        # Conversation context
        self.conversation_history = []
        
        # Conversation starters and follow-up prompts
        self.conversation_starters = [
            "What news headline would you like to discuss today?",
            "I'm ready to analyze a news headline for you. What's on your mind?",
            "Curious about the sentiment behind a recent news story?",
            "Let's explore the nuances of a news headline together."
        ]
        
        self.follow_up_prompts = [
            "Would you like to analyze another headline?",
            "Any other news you'd like me to take a look at?",
            "Is there another headline you're curious about?",
            "Feel free to share another news story."
        ]
    
    def analyze_headline(self, headline):
        """Analyze a headline and generate a detailed response"""
        # Get basic sentiment
        overall_sentiment = self.sentiment_analyzer.predict_sentiment(headline)
        
        # Get aspect-based sentiments
        aspect_results = self.aspect_analyzer.analyze_aspects(headline)
        
        # Generate response
        response = self._generate_response(headline, overall_sentiment, aspect_results)
        
        # Update conversation history
        self.conversation_history.append({
            'headline': headline,
            'sentiment': overall_sentiment,
            'aspects': aspect_results
        })
        
        return response
    
    def _generate_response(self, headline, overall_sentiment, aspect_results):
        """Generate a natural language response based on analysis results"""
        # Sentiment-based opening
        sentiment_openings = {
            'positive': [
                "Great news! ",
                "Here's a positive perspective: ",
                "Some encouraging insights: "
            ],
            'negative': [
                "Let's unpack this challenging headline: ",
                "This headline reveals some concerning aspects: ",
                "Here's a critical analysis: "
            ],
            'neutral': [
                "An interesting headline with balanced implications: ",
                "Let's explore the nuances of this news: ",
                "Here's a balanced view of the headline: "
            ]
        }
        
        # Generate response
        response = random.choice(sentiment_openings[overall_sentiment])
        response += f"Analyzing: '{headline}'\n\n"
        
        # Overall sentiment description
        sentiment_descriptions = {
            'positive': "This headline suggests a positive development.",
            'negative': "This headline indicates some challenging circumstances.",
            'neutral': "This headline presents a balanced or informative perspective."
        }
        response += f"Overall Sentiment: {overall_sentiment.capitalize()} - {sentiment_descriptions[overall_sentiment]}\n\n"
        
        # Aspect-based analysis
        response += "Aspect-based Insights:\n"
        for aspect, sentiment in aspect_results.items():
            if aspect != 'headline' and sentiment != 'neutral':
                aspect_insights = {
                    'government': {
                        'positive': "Positive governmental actions or policies",
                        'negative': "Challenges or criticisms in governance"
                    },
                    'economy': {
                        'positive': "Promising economic indicators",
                        'negative': "Economic challenges or setbacks"
                    },
                    'infrastructure': {
                        'positive': "Improvements in infrastructure",
                        'negative': "Infrastructure-related difficulties"
                    }
                }
                response += f"- {aspect.capitalize()}: {sentiment} - {aspect_insights[aspect][sentiment]}\n"
        
        return response
    
    def start_interactive_session(self):
        """Start an interactive chat session"""
        print("\n" + "="*50)
        print("🌍 South African News Sentiment Chatbot 📰")
        print("="*50)
        
        # Conversation starter
        print("\n" + random.choice(self.conversation_starters))
        
        while True:
            try:
                # Get user input
                headline = input("\n📝 Enter a news headline (or 'quit' to exit): ").strip()
                
                # Exit condition
                if headline.lower() in ['quit', 'exit', 'bye']:
                    print("\nThank you for using the News Sentiment Chatbot. Goodbye!")
                    break
                
                # Analyze headline
                response = self.analyze_headline(headline)
                print("\n📊 Analysis:")
                print(response)
                
                # Follow-up prompt
                print("\n" + random.choice(self.follow_up_prompts))
            
            except Exception as e:
                print(f"\n❌ An error occurred: {e}")
                print("Let's try again.")

def main():
    # Initialize and start chatbot
    chatbot = NewsSentimentChatbot()
    chatbot.start_interactive_session()

if __name__ == "__main__":
    main()