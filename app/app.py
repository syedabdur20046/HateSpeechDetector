"""
SafeNet AI — Flask Web Application
Hate Speech Detection with real-time prediction, confidence scores,
explainability, and CSV export.
Python 3.14 compatible.
"""
from __future__ import annotations

import io
import csv
import json
import pickle
import logging
import datetime
from pathlib import Path

from flask import (
    Flask, request, render_template,
    jsonify, Response, redirect, url_for,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BASE_DIR  = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "safenet-ai-secret-2024"

# ── globals (loaded once at startup) ──────────────────────────────────────
_model      = None
_vectorizer = None
_metadata   = {}
_history: list[dict] = []

LABEL_MAP   = {0: "Hate Speech", 1: "Offensive", 2: "Neutral"}
LABEL_EMOJI = {"Hate Speech": "🚨", "Offensive": "⚠️",  "Neutral": "✅"}
LABEL_COLOR = {"Hate Speech": "danger", "Offensive": "warning", "Neutral": "success"}


def load_model() -> bool:
    global _model, _vectorizer, _metadata
    model_path = MODEL_DIR / "best_model.pkl"
    vec_path   = MODEL_DIR / "vectorizer.pkl"
    meta_path  = MODEL_DIR / "metadata.json"

    if not model_path.exists() or not vec_path.exists():
        logger.warning("Model not found. Run train.py first.")
        return False

    with open(model_path, "rb") as f:
        _model = pickle.load(f)
    with open(vec_path, "rb") as f:
        _vectorizer = pickle.load(f)
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            _metadata = json.load(f)

    logger.info("Model loaded: %s", _metadata.get("best_model", "unknown"))
    return True


def preprocess(text: str) -> str:
    import re
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_text(text: str) -> dict:
    """Returns prediction dict with label, confidence, severity, top words."""
    if _model is None or _vectorizer is None:
        return {"error": "Model not loaded. Run train.py first."}

    cleaned  = preprocess(text)
    features = _vectorizer.transform([cleaned])

    label_id = int(_model.predict(features)[0])
    label    = LABEL_MAP.get(label_id, "Unknown")

    # Confidence — use predict_proba when available, else decision_function
    confidence = 0.0
    all_probs: list[float] = []
    try:
        probs      = _model.predict_proba(features)[0]
        confidence = float(probs[label_id])
        all_probs  = [round(float(p), 4) for p in probs]
    except AttributeError:
        try:
            df_vals    = _model.decision_function(features)[0]
            # Softmax approximation
            e = [2.718 ** v for v in df_vals]
            s = sum(e)
            all_probs  = [round(v / s, 4) for v in e]
            confidence = all_probs[label_id]
        except Exception:
            confidence = 0.85

    # Severity score 0-100
    severity_map = {"Hate Speech": 90, "Offensive": 50, "Neutral": 5}
    base_sev     = severity_map[label]
    severity     = int(base_sev * (0.5 + confidence * 0.5))

    # Top contributing words (TF-IDF feature names)
    top_words: list[str] = []
    try:
        feat_arr     = features.toarray()[0]
        vocab        = _vectorizer.get_feature_names_out()
        top_indices  = feat_arr.argsort()[-8:][::-1]
        top_words    = [str(vocab[i]) for i in top_indices if feat_arr[i] > 0]
    except Exception:
        pass

    # Label probabilities by name
    label_probs: dict[str, float] = {}
    for i, name in LABEL_MAP.items():
        label_probs[name] = all_probs[i] if i < len(all_probs) else 0.0

    result = {
        "text":        text,
        "label":       label,
        "label_id":    label_id,
        "confidence":  round(confidence * 100, 1),
        "severity":    severity,
        "emoji":       LABEL_EMOJI[label],
        "color":       LABEL_COLOR[label],
        "top_words":   top_words,
        "label_probs": label_probs,
        "timestamp":   datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _history.append(result)
    return result


# ── routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    stats = _compute_stats()
    return render_template("index.html",
                           model_name=_metadata.get("best_model", "—"),
                           stats=stats)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or request.form.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    result = predict_text(text)
    return jsonify(result)


@app.route("/history")
def history():
    return render_template("history.html", history=list(reversed(_history[-50:])))


@app.route("/export")
def export_csv():
    if not _history:
        return redirect(url_for("index"))
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["timestamp", "text", "label", "confidence", "severity"])
    writer.writeheader()
    for row in _history:
        writer.writerow({
            "timestamp":  row["timestamp"],
            "text":       row["text"],
            "label":      row["label"],
            "confidence": str(row["confidence"]) + "%",
            "severity":   row["severity"],
        })
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )


@app.route("/dashboard")
def dashboard():
    results = _metadata.get("results", {})
    return render_template("dashboard.html", results=results, history=_history)


@app.route("/api/stats")
def api_stats():
    return jsonify(_compute_stats())


def _compute_stats() -> dict:
    total  = len(_history)
    counts = {"Hate Speech": 0, "Offensive": 0, "Neutral": 0}
    for h in _history:
        label = h.get("label", "Neutral")
        counts[label] = counts.get(label, 0) + 1
    return {"total": total, "counts": counts}


# ── main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_model()
    app.run(debug=True, host="0.0.0.0", port=5000)
