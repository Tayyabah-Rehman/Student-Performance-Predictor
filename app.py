"""
app.py  –  Student Academic Performance Predictor
Flask REST API + static file serving
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib, json, numpy as np, os

BASE = os.path.dirname(__file__)

app   = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# ── Load artefacts ────────────────────────────────────────────────────
model   = joblib.load(os.path.join(BASE, "model/model.pkl"))
scaler  = joblib.load(os.path.join(BASE, "model/scaler.pkl"))
le      = joblib.load(os.path.join(BASE, "model/label_encoder.pkl"))
with open(os.path.join(BASE, "model/meta.json")) as f:
    meta = json.load(f)


# ── Routes ────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        body = request.get_json(force=True)
        features = [
            float(body["study_hours_per_day"]),
            float(body["attendance_percentage"]),
            float(body["assignments_completed"]),
            float(body["previous_semester_marks"]),
            float(body["class_participation"]),
            float(body["sleep_hours"]),
            float(body["extracurricular_activities"]),
        ]
        X = scaler.transform([features])
        pred_enc   = model.predict(X)[0]
        pred_label = le.inverse_transform([pred_enc])[0]

        # probability breakdown
        probs = {}
        if hasattr(model, "predict_proba"):
            raw = model.predict_proba(X)[0]
            probs = {le.inverse_transform([i])[0]: round(float(p)*100, 1)
                     for i, p in enumerate(raw)}
        else:
            probs = {pred_label: 100.0}

        return jsonify({
            "prediction":    pred_label,
            "probabilities": probs,
            "model_used":    meta["best_model"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/meta")
def get_meta():
    return jsonify({
        "test_accuracy":         meta["test_accuracy"],
        "f1_score":              meta["f1_score"],
        "model_comparison":      meta["model_comparison"],
        "feature_importances":   meta["feature_importances"],
        "classes":               meta["classes"],
        "best_model":            meta["best_model"],
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)


# Serve CSV for the dataset preview tab
from flask import send_file
@app.route("/data/student_performance.csv")
def serve_csv():
    return send_file(os.path.join(BASE, "data/student_performance.csv"),
                     mimetype="text/csv")
