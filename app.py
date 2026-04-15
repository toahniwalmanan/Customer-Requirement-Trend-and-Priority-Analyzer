import os
import json
from datetime import datetime
from flask import Flask, render_template, request, send_from_directory, url_for

from predict_demand import analyze_product_demand, load_market_trends
from trend_detection import detect_trends
from visualize import create_dataset_visualizations

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VISUALIZATIONS_DIR = os.path.join(BASE_DIR, "visualizations")
MARKET_TRENDS_PATH = os.path.join(BASE_DIR, "market_trends.json")
PRODUCT_HISTORY_PATH = os.path.join(BASE_DIR, "product_analysis_history.json")

app = Flask(__name__)


def ensure_analysis_assets():
    if not os.path.exists(MARKET_TRENDS_PATH):
        detect_trends()

def load_product_history():
    if not os.path.exists(PRODUCT_HISTORY_PATH):
        return []

    with open(PRODUCT_HISTORY_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def save_product_history(history):
    with open(PRODUCT_HISTORY_PATH, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)


def record_product_analysis(result):
    history = load_product_history()
    history.append(
        {
            "product_name": result["product_name"],
            "demand_score": result["demand_score"],
            "success_score": result["success_score"],
            "prediction": result["prediction"],
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    save_product_history(history)


def build_dashboard_context():
    visualizations, app_summary, sector_summary = create_dataset_visualizations(VISUALIZATIONS_DIR)
    history = load_product_history()
    recent_products = list(reversed(history[-8:]))
    top_apps = app_summary.head(8).to_dict(orient="records")
    top_sectors = sector_summary.head(6).to_dict(orient="records")

    return {
        "visualizations": visualizations,
        "recent_products": recent_products,
        "top_apps": top_apps,
        "top_sectors": top_sectors,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    form_data = {"product_name": "", "product_description": ""}

    if request.method == "POST":
        form_data["product_name"] = request.form.get("product_name", "").strip()
        form_data["product_description"] = request.form.get("product_description", "").strip()

        if not form_data["product_description"]:
            error = "Please enter a short product description so the model has something to analyze."
        else:
            try:
                ensure_analysis_assets()
                trends = load_market_trends()
                result = analyze_product_demand(
                    form_data["product_name"] or "New Product",
                    form_data["product_description"],
                    trends=trends,
                )
                record_product_analysis(result)
            except FileNotFoundError:
                error = "Trend data is missing. Please run the analysis pipeline to generate market_trends.json."

    return render_template("index.html", result=result, error=error, form_data=form_data)


@app.route("/dashboard")
def dashboard():
    context = build_dashboard_context()
    return render_template("dashboard.html", **context)


@app.route("/visualizations/<path:filename>")
def visualization_file(filename):
    return send_from_directory(VISUALIZATIONS_DIR, filename)


@app.context_processor
def inject_navigation():
    return {"dashboard_url": url_for("dashboard"), "home_url": url_for("index")}


if __name__ == "__main__":
    app.run(debug=True)
