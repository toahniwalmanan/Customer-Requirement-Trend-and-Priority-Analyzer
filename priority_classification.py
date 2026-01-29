import joblib
from preprocess import preprocess_text
import re

# Load the trained model and vectorizer
model = joblib.load('sentiment_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

HIGH_PRIORITY_KEYWORDS = ["crash", "error", "bug", "issue", "problem", "urgent", "critical", "fail", "broken", "unable", "stuck"]
MEDIUM_PRIORITY_KEYWORDS = ["add", "feature", "request", "suggestion", "slow", "performance"]

def classify_priority(review_text):
    # Preprocess the review text
    preprocessed_text = preprocess_text(review_text)
    
    # Predict the sentiment
    text_vector = vectorizer.transform([preprocessed_text])
    sentiment = model.predict(text_vector)[0]
    
    # Classify priority based on rules
    if sentiment == 'negative':
        for keyword in HIGH_PRIORITY_KEYWORDS:
            if re.search(r'\b' + keyword + r'\b', review_text.lower()):
                return 'High'
        return 'Medium'
    elif sentiment == 'neutral':
        for keyword in MEDIUM_PRIORITY_KEYWORDS:
            if re.search(r'\b' + keyword + r'\b', review_text.lower()):
                return 'Medium'
        return 'Low'
    else: # positive sentiment
        return 'Low'

if __name__ == '__main__':
    sample_reviews = [
        "This app is constantly crashing, it's a huge problem!",
        "The latest update is really slow and my battery drains faster.",
        "It would be great if you could add a dark mode feature.",
        "I love this app, it works perfectly!",
        "The app is okay, but it could be better."
    ]
    
    for review in sample_reviews:
        priority = classify_priority(review)
        print(f"Review: '{review}'\nPriority: {priority}\n")
