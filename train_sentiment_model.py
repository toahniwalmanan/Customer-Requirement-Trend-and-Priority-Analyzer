import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from preprocess import preprocess_text, download_nltk_data

def train_and_evaluate():
    # Download NLTK data if not present
    download_nltk_data()

    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv("App_Review_Labelled.csv")

    # Filter out 'unknown' sentiment labels
    print("Filtering out 'unknown' labels...")
    df = df[df['sentiment_label'] != 'unknown']

    # Preprocess the text data
    print("Preprocessing text data...")
    df['processed_text'] = df['text'].apply(preprocess_text)

    # Feature Extraction using TF-IDF
    print("Extracting features using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['processed_text'])
    y = df['sentiment_label']

    # Split data into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Logistic Regression model
    print("Training the Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate the model
    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save the model and vectorizer
    print("Saving the model and vectorizer...")
    joblib.dump(model, 'sentiment_model.joblib')
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
    print("Model and vectorizer saved successfully.")

if __name__ == '__main__':
    train_and_evaluate()