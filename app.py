"""
Flask API and Sentiment Analysis Dashboard.
- GET /  -> Dashboard UI
- POST /api/predict -> JSON { "text": "..." } -> sentiment + confidence scores
"""
import os
from flask import Flask, request, jsonify, render_template

from inference import predict

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Missing or empty 'text' field"}), 400
    try:
        result = predict(text)
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)
