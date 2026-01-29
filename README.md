# Customer Feedback Analysis and Product Demand Prediction

This project uses Natural Language Processing (NLP) and Machine Learning to analyze customer feedback from app reviews. It extracts meaningful insights, detects trending requirements, classifies priority levels, and predicts the market fit of new product ideas.

## 1. Project Overview

In modern software development, customer requirements change frequently and arrive in large volumes through user feedback, reviews, and support tickets. Manually analyzing this data is time-consuming and error-prone. This project provides an automated system to analyze this feedback and provide data-driven insights to support requirement engineering and project planning.

### Key Objectives:
- Extract meaningful information from textual feedback.
- Perform sentiment analysis on customer requirements.
- Identify trending features using keyword frequency.
- Classify requirement priority (High / Medium / Low).
- Predict the market demand for a new product concept.
- Visualize trends for managerial decision-making.

## 2. Data

This project uses two datasets:
- **`App_Review_Labelled.csv`**: Contains 15,000 app reviews with pre-labeled sentiment ('positive', 'negative', 'neutral'). This dataset is used to train the sentiment analysis model.
- **`Review_Dataset.csv`**: A dataset of app reviews without sentiment labels, used to demonstrate the full analysis pipeline on raw, unseen data.

## 3. Step-by-Step Procedure

The project is structured into a series of modules that work together in a pipeline:

**Step 1: Environment Setup**
- A Python virtual environment is created to manage dependencies.
- All required libraries (e.g., `pandas`, `nltk`, `scikit-learn`, `matplotlib`) are installed via `pip`.

**Step 2: Text Preprocessing**
- A text preprocessing module (`preprocess.py`) cleans the raw review text by performing:
    - Lowercasing
    - Punctuation removal
    - Tokenization (splitting text into words)
    - Stop-word removal (removing common words like 'the', 'is', 'a')
    - Stemming (reducing words to their root form, e.g., 'running' -> 'run')

**Step 3: Sentiment Analysis**
- A sentiment analysis model is trained to classify reviews as 'positive', 'negative', or 'neutral'.
- The `App_Review_Labelled.csv` dataset is used for training.
- **TF-IDF (Term Frequency-Inverse Document Frequency)** is used to convert the text into numerical features.
- A **Logistic Regression** model is trained on these features.
- The trained model (`sentiment_model.joblib`) and TF-IDF vectorizer (`tfidf_vectorizer.joblib`) are saved to disk.

**Step 4: Trend Detection & Analysis**
- The system analyzes the frequency of keywords and phrases (n-grams) to identify what users are talking about most.
- It also calculates the average sentiment associated with each trend, providing context to the keywords (e.g., is 'update' usually mentioned in a positive or negative context?).
- These trends are saved in `market_trends.json`.

**Step 5: Priority Classification**
- A rule-based system (`priority_classification.py`) assigns a priority level (High, Medium, or Low) to each review based on:
    - **Sentiment:** Negative reviews are prioritized.
    - **Keywords:** The presence of critical keywords (e.g., 'crash', 'bug', 'urgent') increases the priority.

**Step 6: Visualization & Reporting**
- The `visualize.py` script generates a series of plots to provide a clear visual summary of the analysis. These are saved in the `visualizations/` directory and include:
    - A pie chart of the sentiment distribution.
    - Bar charts of the top keywords and bigrams.
    - A word cloud for a quick overview of the most discussed topics.

**Step 7: Product Demand Prediction**
- A predictive feature (`predict_demand.py`) estimates the market fit of a new product idea.
- It scores the user's product description against the pre-analyzed market trends from `market_trends.json`.
- It provides a "Market Fit Score" and a qualitative analysis of how the product aligns with customer needs and pain points.

## 4. How to Use the System

### Setup
First, install the required dependencies:
```bash
# Activate the virtual environment
source venv/bin/activate  # On Linux/macOS
.\venv\Scripts\Activate.ps1 # On Windows

# Install libraries
pip install -r requirements.txt
```

### Feature 1: Analyze a Review Dataset
To run the full analysis pipeline on a CSV file of reviews, use the `main.py` script. The input CSV must have a column named `review` or `text`.

```bash
python main.py "path/to/your/reviews.csv"
```
This will:
1. Generate `analysis_report.csv` with the sentiment and priority for each review.
2. Create and save plots in the `visualizations/` directory.
3. Print a summary of the analysis to the console.

### Feature 2: Predict Demand for a New Product
To get a market fit analysis for a new product idea, run the `predict_demand.py` script.

```bash
python predict_demand.py
```
The script will prompt you to enter a description of your product. It will then output a Market Fit Score and a report.

## 5. File Descriptions

- **`main.py`**: The main script to run the full analysis pipeline on a review dataset.
- **`predict_demand.py`**: The script to predict the market fit of a new product idea.
- **`analysis_report.csv`**: The output report from `main.py`.
- **`market_trends.json`**: A file containing the top keywords and bigrams and their associated sentiment scores.
- **`visualizations/`**: The directory where all plots are saved.
- **`sentiment_model.joblib`**: The trained sentiment analysis model.
- **`tfidf_vectorizer.joblib`**: The saved TF-IDF vectorizer.
- **`preprocess.py`**: The module for text preprocessing.
- **`train_sentiment_model.py`**: The script to train the sentiment analysis model.
- **`trend_detection.py`**: The module for trend detection and analysis.
- **`priority_classification.py`**: The module for priority classification.
- **`visualize.py`**: The module for generating visualizations.
- **`requirements.txt`**: A list of all Python dependencies.
- **`nltk_data/`**: A directory containing NLTK data for preprocessing.
- **`venv/`**: The Python virtual environment.

## 6. Project Conclusion

This project successfully demonstrates the application of NLP and Machine Learning techniques to automate the analysis of customer feedback. By integrating sentiment analysis, trend detection, priority classification, and product demand prediction, the system provides valuable, data-driven insights that can significantly enhance requirement engineering, product development, and strategic decision-making in software projects.

**SEPM Concepts Covered:**

*   **Requirement Engineering:** Automated analysis and extraction of requirements from unstructured feedback.
*   **Change Management:** Trend detection identifies evolving user needs and shifts in sentiment over time.
*   **Decision Support:** Priority classification and sentiment analysis provide data for informed decision-making on feature development.
*   **Risk Reduction:** Early detection of critical issues (e.g., bugs, crashes) through high-priority classification.
*   **Project Planning:** Feature prioritization based on sentiment, frequency, and market demand prediction.

This tool helps development teams to quickly understand user needs, identify critical issues, assess the potential of new features or products, and plan projects more effectively, ultimately leading to more user-centric and successful software solutions.