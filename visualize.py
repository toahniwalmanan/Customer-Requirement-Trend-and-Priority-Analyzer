import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from preprocess import preprocess_text, download_nltk_data
import os

def create_visualizations():
    # Download NLTK data if not present
    download_nltk_data()

    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = pd.read_csv("App_Review_Labelled.csv")
    df = df[df['sentiment_label'] != 'unknown']
    df['processed_text'] = df['text'].apply(preprocess_text)

    # Create output directory
    output_dir = "visualizations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Sentiment Distribution Pie Chart
    print("Generating sentiment distribution pie chart...")
    plt.figure(figsize=(8, 8))
    df['sentiment_label'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette("viridis", 3))
    plt.title('Sentiment Distribution of App Reviews')
    plt.ylabel('')
    plt.savefig(os.path.join(output_dir, "sentiment_distribution.png"))
    plt.close()

    # 2. Top Keywords Bar Chart
    print("Generating top keywords bar chart...")
    vectorizer = CountVectorizer(max_features=10)
    X = vectorizer.fit_transform(df['processed_text'])
    word_counts = X.toarray().sum(axis=0)
    word_freq = pd.DataFrame({'word': vectorizer.get_feature_names_out(), 'frequency': word_counts})
    top_keywords = word_freq.sort_values(by='frequency', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='frequency', y='word', data=top_keywords, palette='viridis')
    plt.title('Top 10 Most Frequent Keywords')
    plt.xlabel('Frequency')
    plt.ylabel('Keyword')
    plt.savefig(os.path.join(output_dir, "top_keywords.png"))
    plt.close()

    # 3. Top Bigrams Bar Chart
    print("Generating top bigrams bar chart...")
    bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=10)
    X_bigram = bigram_vectorizer.fit_transform(df['processed_text'])
    bigram_counts = X_bigram.toarray().sum(axis=0)
    bigram_freq = pd.DataFrame({'bigram': bigram_vectorizer.get_feature_names_out(), 'frequency': bigram_counts})
    top_bigrams = bigram_freq.sort_values(by='frequency', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='frequency', y='bigram', data=top_bigrams, palette='viridis')
    plt.title('Top 10 Most Frequent Bigrams')
    plt.xlabel('Frequency')
    plt.ylabel('Bigram')
    plt.savefig(os.path.join(output_dir, "top_bigrams.png"))
    plt.close()


    # 4. Word Cloud
    print("Generating word cloud...")
    all_text = ' '.join(df['processed_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    
    plt.figure(figsize=(15, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of App Reviews')
    plt.savefig(os.path.join(output_dir, "word_cloud.png"))
    plt.close()

    print(f"\nVisualizations saved in the '{output_dir}' directory.")

if __name__ == '__main__':
    create_visualizations()
