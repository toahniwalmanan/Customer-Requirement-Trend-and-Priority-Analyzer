import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import os
import pandas as pd

# Define and set the NLTK data directory at the module level
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

def download_nltk_data():
    # Download 'punkt' and 'stopwords' if not already downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading 'punkt' tokenizer...")
        nltk.download('punkt', download_dir=nltk_data_dir)

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading 'stopwords' corpus...")
        nltk.download('stopwords', download_dir=nltk_data_dir)

def preprocess_text(text):
    if pd.isna(text):
        return ""

    if not isinstance(text, str):
        text = str(text)

    # Lowercasing
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenization
    tokens = word_tokenize(text)

    # Stop-word removal
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

    return " ".join(stemmed_tokens)

if __name__ == '__main__':
    download_nltk_data()

    sample_review = "Some systems just stop. Like today, it didn't record my sleep. Not even a little."
    print(f"Original review: {sample_review}")

    preprocessed_review = preprocess_text(sample_review)
    print(f"Preprocessed review: {preprocessed_review}")
