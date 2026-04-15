import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _compute_scores(frame):
    positive = frame.get("positive", 0)
    neutral = frame.get("neutral", 0)
    negative = frame.get("negative", 0)
    total = positive + neutral + negative

    if total.eq(0).any():
        total = total.replace(0, 1)

    positive_share = positive / total
    neutral_share = neutral / total
    negative_share = negative / total

    demand_score = ((positive_share * 0.8) + (neutral_share * 0.6) + (negative_share * 1.0)) * 10
    success_score = ((positive_share * 1.0) + (neutral_share * 0.65) + (negative_share * 0.15)) * 10

    return demand_score.round(2), success_score.round(2)


def _prepare_dataset_aggregates(dataset_path="App_Review_Labelled.csv"):
    df = pd.read_csv(dataset_path)
    df = df[df["sentiment_label"] != "unknown"].copy()

    app_summary = (
        df.groupby(["app_id", "sector", "sentiment_label"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for label in ["positive", "neutral", "negative"]:
        if label not in app_summary.columns:
            app_summary[label] = 0

    app_summary["review_count"] = app_summary["positive"] + app_summary["neutral"] + app_summary["negative"]
    app_summary["demand_score"], app_summary["success_score"] = _compute_scores(app_summary)
    app_summary["app_label"] = app_summary["app_id"].str.rsplit(".", n=1).str[-1].str.slice(0, 22)
    app_summary = app_summary.sort_values(["demand_score", "success_score"], ascending=False)

    sector_summary = (
        df.groupby(["sector", "sentiment_label"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for label in ["positive", "neutral", "negative"]:
        if label not in sector_summary.columns:
            sector_summary[label] = 0

    sector_summary["review_count"] = (
        sector_summary["positive"] + sector_summary["neutral"] + sector_summary["negative"]
    )
    sector_summary["demand_score"], sector_summary["success_score"] = _compute_scores(sector_summary)
    sector_summary = sector_summary.sort_values(["demand_score", "success_score"], ascending=False)

    return app_summary, sector_summary


def create_dataset_visualizations(output_dir="visualizations", dataset_path="App_Review_Labelled.csv"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    app_summary, sector_summary = _prepare_dataset_aggregates(dataset_path)
    sns.set_theme(style="whitegrid")

    top_apps = app_summary.head(10).copy()
    comparison_df = top_apps.melt(
        id_vars=["app_label"],
        value_vars=["demand_score", "success_score"],
        var_name="metric",
        value_name="score",
    )
    comparison_df["metric"] = comparison_df["metric"].map(
        {"demand_score": "Demand Score", "success_score": "Success Score"}
    )

    top_apps_path = os.path.join(output_dir, "dataset_top_apps_scores.png")
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=comparison_df,
        x="score",
        y="app_label",
        hue="metric",
        palette=["#b45309", "#0f766e"],
    )
    plt.xlim(0, 10)
    plt.xlabel("Score")
    plt.ylabel("App")
    plt.title("Top Apps by Demand and Success Scores")
    plt.tight_layout()
    plt.savefig(top_apps_path)
    plt.close()

    map_path = os.path.join(output_dir, "dataset_app_positioning_map.png")
    plt.figure(figsize=(11, 7))
    scatter = sns.scatterplot(
        data=app_summary,
        x="demand_score",
        y="success_score",
        hue="sector",
        palette="tab10",
        s=140,
    )
    for _, row in app_summary.iterrows():
        scatter.text(
            row["demand_score"] + 0.04,
            row["success_score"] + 0.04,
            row["app_label"],
            fontsize=8,
        )
    plt.axvline(5, linestyle="--", color="#bdbdbd", linewidth=1)
    plt.axhline(5, linestyle="--", color="#bdbdbd", linewidth=1)
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.xlabel("Demand Score")
    plt.ylabel("Success Score")
    plt.title("App Positioning Map by Dataset Feedback")
    plt.tight_layout()
    plt.savefig(map_path)
    plt.close()

    sector_df = sector_summary.melt(
        id_vars=["sector"],
        value_vars=["demand_score", "success_score"],
        var_name="metric",
        value_name="score",
    )
    sector_df["metric"] = sector_df["metric"].map(
        {"demand_score": "Demand Score", "success_score": "Success Score"}
    )

    sector_path = os.path.join(output_dir, "dataset_sector_scores.png")
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=sector_df,
        x="score",
        y="sector",
        hue="metric",
        palette=["#d97706", "#0f766e"],
    )
    plt.xlim(0, 10)
    plt.xlabel("Average Score")
    plt.ylabel("Sector")
    plt.title("Sector-Level Demand and Success Scores")
    plt.tight_layout()
    plt.savefig(sector_path)
    plt.close()

    visualizations = [
        {
            "title": "Top Apps Score Comparison",
            "description": "Demand and success scores derived from labeled feedback for the strongest apps in the dataset.",
            "filename": "dataset_top_apps_scores.png",
        },
        {
            "title": "App Positioning Map",
            "description": "A portfolio view of how apps sit across high-demand and high-success zones.",
            "filename": "dataset_app_positioning_map.png",
        },
        {
            "title": "Sector Score Overview",
            "description": "Average demand and success scores across sectors represented in the dataset.",
            "filename": "dataset_sector_scores.png",
        },
    ]

    return visualizations, app_summary, sector_summary


def create_visualizations():
    visualizations, _, _ = create_dataset_visualizations()
    return visualizations


if __name__ == "__main__":
    create_dataset_visualizations()
