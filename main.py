import pandas as pd
import argparse
from preprocess import preprocess_text, download_nltk_data
from priority_classification import classify_priority
from trend_detection import detect_trends
from visualize import create_visualizations
import joblib

def analyze_reviews(file_path):
    """
    Main function to run the full analysis pipeline on a CSV file of reviews.
    """
    # Ensure NLTK data is available
    download_nltk_data()
    
    # Load the trained sentiment model and vectorizer
    sentiment_model = joblib.load('sentiment_model.joblib')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

    # 1. Load and preprocess the data
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    # Assuming the review text is in a column named 'text' or 'review'
    if 'text' in df.columns:
        text_column = 'text'
    elif 'review' in df.columns:
        text_column = 'review'
    else:
        print("Error: The CSV must contain a 'text' or 'review' column.")
        return

    print("Preprocessing text data...")
    df['processed_text'] = df[text_column].apply(preprocess_text)

    # 2. Perform Analysis (Sentiment and Priority)
    print("Analyzing sentiment and priority...")
    
    def get_sentiment(text):
        vector = tfidf_vectorizer.transform([text])
        return sentiment_model.predict(vector)[0]
        
    df['sentiment'] = df['processed_text'].apply(get_sentiment)
    df['priority'] = df[text_column].apply(classify_priority)


    # 3. Generate Report
    print("Generating analysis report...")
    report_df = df[[text_column, 'processed_text', 'sentiment', 'priority']]
    report_df.to_csv("analysis_report.csv", index=False)
    print("Report saved to analysis_report.csv")

    # 4. Print Summary
    print("\n--- Analysis Summary ---")
    print("\nSentiment Distribution:")
    print(df['sentiment'].value_counts(normalize=True).to_string())
    
    print("\nPriority Distribution:")
    print(df['priority'].value_counts(normalize=True).to_string())

    # 5. Detect and Display Trends
    detect_trends()

    # 6. Generate Visualizations
    print("\nGenerating visualizations...")
    create_visualizations()
    
    print("\n--- Analysis Complete ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze customer reviews from a CSV file.')
    parser.add_argument('file_path', type=str, help='The path to the input CSV file.')
    args = parser.parse_args()
    
    analyze_reviews(args.file_path)
