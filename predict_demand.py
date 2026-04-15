import json
import re

import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import preprocess_text, download_nltk_data

REVIEW_INDEX = None


def load_market_trends():
    with open("market_trends.json", "r") as f:
        return json.load(f)


def load_review_index():
    global REVIEW_INDEX

    if REVIEW_INDEX is not None:
        return REVIEW_INDEX

    download_nltk_data()

    df = pd.read_csv("App_Review_Labelled.csv")
    df = df[df["sentiment_label"] != "unknown"].copy()

    text_column = "text" if "text" in df.columns else "review"
    df["processed_text"] = df[text_column].fillna("").apply(preprocess_text)

    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    review_matrix = vectorizer.transform(df["processed_text"])

    REVIEW_INDEX = {
        "df": df,
        "text_column": text_column,
        "vectorizer": vectorizer,
        "review_matrix": review_matrix,
    }
    return REVIEW_INDEX


def extract_matching_trends(processed_description, trends):
    all_trends = trends.get("keywords", []) + trends.get("bigrams", [])
    matches = []

    for trend in all_trends:
        term = trend["term"]
        if re.search(r"\b" + re.escape(term) + r"\b", processed_description):
            matches.append(
                {
                    "term": term,
                    "sentiment": trend["sentiment"],
                    "frequency": trend["frequency"],
                }
            )

    return sorted(matches, key=lambda item: item["frequency"], reverse=True)[:8]


def analyze_product_demand(product_name, product_description, trends=None):
    download_nltk_data()

    if trends is None:
        trends = load_market_trends()

    review_index = load_review_index()
    processed_description = preprocess_text(f"{product_name} {product_description}")
    product_vector = review_index["vectorizer"].transform([processed_description])
    similarities = cosine_similarity(product_vector, review_index["review_matrix"]).ravel()

    similarity_cutoff = 0.15
    ranked_indices = similarities.argsort()[::-1]
    relevant_mask = similarities >= similarity_cutoff
    if relevant_mask.sum() < 25:
        top_indices = ranked_indices[:60]
    else:
        top_indices = ranked_indices[relevant_mask[ranked_indices]][:120]

    relevant_reviews = review_index["df"].iloc[top_indices].copy()
    relevant_reviews["similarity"] = similarities[top_indices]
    relevant_reviews = relevant_reviews[relevant_reviews["similarity"] > 0]

    if relevant_reviews.empty:
        relevant_reviews = review_index["df"].iloc[similarities.argsort()[::-1][:50]].copy()
        relevant_reviews["similarity"] = similarities[similarities.argsort()[::-1][:50]]

    weighted_sentiment = (
        relevant_reviews.groupby("sentiment_label")["similarity"].sum().to_dict()
    )
    sentiment_counts = relevant_reviews["sentiment_label"].value_counts().to_dict()

    positive_weight = weighted_sentiment.get("positive", 0.0)
    neutral_weight = weighted_sentiment.get("neutral", 0.0)
    negative_weight = weighted_sentiment.get("negative", 0.0)
    total_weight = positive_weight + neutral_weight + negative_weight

    if total_weight == 0:
        demand_score = 0.0
        success_score = 0.0
        positive_share = neutral_share = negative_share = 0.0
    else:
        positive_share = positive_weight / total_weight
        neutral_share = neutral_weight / total_weight
        negative_share = negative_weight / total_weight
        average_similarity = float(relevant_reviews["similarity"].mean())
        similarity_strength = min(1.0, average_similarity / 0.28)
        support_strength = min(1.0, len(relevant_reviews) / 80)
        demand_base = ((positive_share * 0.8) + (neutral_share * 0.6) + (negative_share * 1.0)) * 10
        demand_confidence = (similarity_strength * 0.55) + (support_strength * 0.45)
        demand_score = demand_base * demand_confidence

        success_base = ((positive_share * 1.0) + (neutral_share * 0.65) + (negative_share * 0.15)) * 10
        success_confidence = (similarity_strength * 0.65) + (support_strength * 0.35)
        success_score = success_base * success_confidence

    if demand_score >= 7 and success_score >= 7:
        prediction = "This looks like a strong opportunity: the problem space appears active, and similar feedback suggests users may respond well to a solid implementation."
    elif demand_score >= 7 and success_score < 6:
        prediction = "This problem area appears to have clear demand, but similar feedback is more frustrated than satisfied, so quality, reliability, and execution will make the difference."
    elif demand_score < 5 and success_score >= 6:
        prediction = "The feedback around similar ideas is fairly positive, but the overall demand signal is narrower, so this may fit a smaller or more niche audience."
    else:
        prediction = "The signal is mixed. There is some overlap with real customer feedback, but both demand and likely success would benefit from a sharper value proposition."

    sentiment_breakdown = [
        {
            "label": "Positive",
            "count": sentiment_counts.get("positive", 0),
            "percentage": round(positive_share * 100, 1),
        },
        {
            "label": "Neutral",
            "count": sentiment_counts.get("neutral", 0),
            "percentage": round(neutral_share * 100, 1),
        },
        {
            "label": "Negative",
            "count": sentiment_counts.get("negative", 0),
            "percentage": round(negative_share * 100, 1),
        },
    ]

    matching_trends = extract_matching_trends(processed_description, trends)
    insights = []
    for item in sentiment_breakdown:
        insights.append(
            f"{item['label']} feedback accounts for {item['percentage']}% of the most similar reviews."
        )

    if matching_trends:
        insights.append(
            "Recurring market terms in this idea include "
            + ", ".join(f"'{item['term']}'" for item in matching_trends[:4])
            + "."
        )

    sample_reviews = []
    text_column = review_index["text_column"]
    for _, row in relevant_reviews.head(3).iterrows():
        sample_reviews.append(
            {
                "text": str(row[text_column])[:220],
                "sentiment": row["sentiment_label"].title(),
                "similarity": round(float(row["similarity"]), 3),
            }
        )

    return {
        "product_name": product_name,
        "product_description": product_description,
        "processed_description": processed_description,
        "demand_score": round(float(demand_score), 2),
        "success_score": round(float(success_score), 2),
        "prediction": prediction,
        "matched_trends": matching_trends,
        "insights": insights,
        "sentiment_breakdown": sentiment_breakdown,
        "sample_reviews": sample_reviews,
        "matched_review_count": int(len(relevant_reviews)),
    }


def predict_product_demand():
    try:
        trends = load_market_trends()
    except FileNotFoundError:
        print("Error: 'market_trends.json' not found. Please run 'trend_detection.py' first.")
        return

    print("\n--- New Product Demand Prediction ---")
    product_name = input("Enter the product or feature name: ").strip() or "New Product"
    product_description = input("Enter a description of your new product or feature: ")
    result = analyze_product_demand(product_name, product_description, trends=trends)

    print("\n--- Market Fit Analysis Report ---")
    print(f"Product: '{result['product_name']}'")
    print(f"Product Description: '{result['product_description']}'")
    print(f"\nDemand Score: {result['demand_score']:.2f} / 10")
    print(f"Success Score: {result['success_score']:.2f} / 10")
    print(f"Prediction: {result['prediction']}")
    print(f"Matched Reviews Analyzed: {result['matched_review_count']}")

    print("\nSentiment Breakdown:")
    for item in result["sentiment_breakdown"]:
        print(f"  - {item['label']}: {item['count']} reviews ({item['percentage']}%)")

    if result["sample_reviews"]:
        print("\nMost Similar Reviews:")
        for review in result["sample_reviews"]:
            print(f"  - [{review['sentiment']}] ({review['similarity']}) {review['text']}")


if __name__ == "__main__":
    predict_product_demand()
