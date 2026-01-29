import json
from preprocess import preprocess_text, download_nltk_data
import re

def predict_product_demand():
    # Download NLTK data if not present
    download_nltk_data()

    # Load market trends
    try:
        with open('market_trends.json', 'r') as f:
            trends = json.load(f)
    except FileNotFoundError:
        print("Error: 'market_trends.json' not found. Please run 'trend_detection.py' first.")
        return

    # Get user input
    print("\n--- New Product Demand Prediction ---")
    product_description = input("Enter a description of your new product or feature: ")

    # Preprocess the product description
    processed_description = preprocess_text(product_description)
    
    # --- Scoring Logic ---
    score = 0
    max_score = 0
    matched_trends = []

    all_trends = trends['keywords'] + trends['bigrams']

    for trend in all_trends:
        term = trend['term']
        term_sentiment = trend['sentiment']
        term_frequency = trend['frequency']
        
        # We give more weight to more frequent terms
        weight = term_frequency / trends['keywords'][0]['frequency'] # Normalize by the top keyword frequency
        
        if re.search(r'\b' + term + r'\b', processed_description):
            # If the product feature aligns with a positive trend, increase score
            # If it aligns with a negative trend, it means it's addressing a pain point, so also increase score
            # The magnitude of the sentiment determines the points
            points = abs(term_sentiment) * weight * 10
            score += points
            
            matched_trends.append({
                "term": term,
                "sentiment": term_sentiment,
                "points": points
            })
        
        # The max score is the sum of all possible points
        max_score += abs(term_sentiment) * weight * 10
        
    # Normalize the score to be out of 10
    market_fit_score = (score / max_score) * 10 if max_score > 0 else 0


    # --- Generate Report ---
    print("\n--- Market Fit Analysis Report ---")
    print(f"Product Description: '{product_description}'")
    print(f"\nMarket Fit Score: {market_fit_score:.2f} / 10")
    
    if market_fit_score >= 7:
        print("Prediction: This product shows a strong alignment with current market trends and addresses key customer talking points.")
    elif market_fit_score >= 4:
        print("Prediction: This product shows some alignment with market trends, but could be improved by addressing more key customer needs.")
    else:
        print("Prediction: This product does not seem to align well with current market trends. Consider re-evaluating the features.")

    if matched_trends:
        print("\nAnalysis Breakdown:")
        for match in matched_trends:
            if match['sentiment'] > 0.2:
                print(f"  - Your product aligns with the positive trend of '{match['term']}'.")
            elif match['sentiment'] < -0.2:
                print(f"  - Your product addresses a common pain point related to '{match['term']}'.")
            else:
                print(f"  - Your product mentions '{match['term']}', which is a neutral but frequent topic.")

if __name__ == '__main__':
    predict_product_demand()
