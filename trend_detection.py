import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from preprocess import preprocess_text, download_nltk_data
import joblib
import json

def analyze_and_save_trends():
    # Download NLTK data if not present
    download_nltk_data()

    # Load the dataset and sentiment model
    print("Loading dataset and sentiment model...")
    df = pd.read_csv("App_Review_Labelled.csv")
    df = df[df['sentiment_label'] != 'unknown']
    
    model = joblib.load('sentiment_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')

    # Preprocess text
    print("Preprocessing text data...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Map sentiment to a numerical score
    sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    df['sentiment_score'] = df['sentiment_label'].map(sentiment_map)

    # Get top keywords
    print("Analyzing keyword trends...")
    keyword_vectorizer = CountVectorizer(max_features=100)
    keyword_vectorizer.fit(df['processed_text'])
    top_keywords = keyword_vectorizer.get_feature_names_out()

    # Calculate sentiment for each keyword
    keyword_trends = []
    for keyword in top_keywords:
        keyword_reviews = df[df['processed_text'].str.contains(r'\b' + keyword + r'\b')]
        if not keyword_reviews.empty:
            avg_sentiment = keyword_reviews['sentiment_score'].mean()
            frequency = len(keyword_reviews)
            keyword_trends.append({'term': keyword, 'frequency': frequency, 'sentiment': avg_sentiment})

    # Get top bigrams
    print("Analyzing bigram trends...")
    bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=100)
    bigram_vectorizer.fit(df['processed_text'])
    top_bigrams = bigram_vectorizer.get_feature_names_out()

    # Calculate sentiment for each bigram
    bigram_trends = []
    for bigram in top_bigrams:
        bigram_reviews = df[df['processed_text'].str.contains(r'\b' + bigram + r'\b')]
        if not bigram_reviews.empty:
            avg_sentiment = bigram_reviews['sentiment_score'].mean()
            frequency = len(bigram_reviews)
            bigram_trends.append({'term': bigram, 'frequency': frequency, 'sentiment': avg_sentiment})
            
    # Sort trends by frequency
    keyword_trends = sorted(keyword_trends, key=lambda x: x['frequency'], reverse=True)
    bigram_trends = sorted(bigram_trends, key=lambda x: x['frequency'], reverse=True)

    # Save trends to JSON
    trends_data = {
        'keywords': keyword_trends[:20], # Save top 20
        'bigrams': bigram_trends[:20]
    }
    
    with open('market_trends.json', 'w') as f:
        json.dump(trends_data, f, indent=4)
        
    print("\nMarket trends saved to market_trends.json")
    print("\nTop 5 Keyword Trends:")
    for trend in keyword_trends[:5]:
        print(f"  - Term: {trend['term']}, Frequency: {trend['frequency']}, Sentiment: {trend['sentiment']:.2f}")

    print("\nTop 5 Bigram Trends:")
    for trend in bigram_trends[:5]:
        print(f"  - Term: {trend['term']}, Frequency: {trend['frequency']}, Sentiment: {trend['sentiment']:.2f}")

def detect_trends():
    """
    Backward-compatible entry point used by main.py.
    """
    analyze_and_save_trends()

if __name__ == '__main__':
    analyze_and_save_trends()
