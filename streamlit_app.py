"""
SafeNet AI — Streamlit Dashboard (Alternative to Flask)
Run: streamlit run streamlit_app.py
Python 3.14 compatible.
"""
from __future__ import annotations

import re
import sys
import json
import pickle
import logging
from pathlib import Path

BASE_DIR  = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"

# ── Streamlit import guard ────────────────────────────────
try:
    import streamlit as st
except ImportError:
    print("Install streamlit:  pip install streamlit")
    sys.exit(1)

LABEL_MAP   = {0: "Hate Speech", 1: "Offensive", 2: "Neutral"}
LABEL_COLOR = {"Hate Speech": "🔴", "Offensive": "🟡", "Neutral": "🟢"}

# ── page config ──────────────────────────────────────────
st.set_page_config(
    page_title="SafeNet AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── load model (cached) ──────────────────────────────────
@st.cache_resource
def load_model():
    model_path = MODEL_DIR / "best_model.pkl"
    vec_path   = MODEL_DIR / "vectorizer.pkl"
    meta_path  = MODEL_DIR / "metadata.json"
    if not model_path.exists():
        return None, None, {}
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
    meta = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    return model, vectorizer, meta


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def predict(text: str, model, vectorizer) -> dict:
    cleaned   = preprocess(text)
    features  = vectorizer.transform([cleaned])
    label_id  = int(model.predict(features)[0])
    label     = LABEL_MAP[label_id]
    confidence = 0.0
    all_probs: list[float] = []
    try:
        probs = model.predict_proba(features)[0]
        confidence = float(probs[label_id])
        all_probs  = [float(p) for p in probs]
    except AttributeError:
        confidence = 0.85
        all_probs  = [0.0, 0.0, 0.0]
        all_probs[label_id] = confidence

    severity_map = {"Hate Speech": 90, "Offensive": 50, "Neutral": 5}
    severity = int(severity_map[label] * (0.5 + confidence * 0.5))

    top_words: list[str] = []
    try:
        feat_arr    = features.toarray()[0]
        vocab       = vectorizer.get_feature_names_out()
        top_indices = feat_arr.argsort()[-8:][::-1]
        top_words   = [str(vocab[i]) for i in top_indices if feat_arr[i] > 0]
    except Exception:
        pass

    return {
        "label": label, "label_id": label_id,
        "confidence": round(confidence * 100, 1),
        "severity": severity,
        "all_probs": all_probs,
        "top_words": top_words,
    }


# ── UI ───────────────────────────────────────────────────
model, vectorizer, meta = load_model()

# Sidebar
with st.sidebar:
    st.markdown("## 🛡️ SafeNet AI")
    st.markdown("Hate Speech Detection using NLP & ML")
    st.divider()
    if meta:
        st.markdown("**Best Model**")
        st.info(meta.get("best_model", "—"))
        st.markdown("**Model Results**")
        results = meta.get("results", {})
        for mname, mvals in results.items():
            with st.expander(mname):
                for k, v in mvals.items():
                    st.metric(k, str(round(v, 4)))
    else:
        st.warning("Run train.py first to load model results.")
    st.divider()
    st.markdown("**Quick Examples**")
    if st.button("Hate example"):
        st.session_state["example"] = "I hate those people, they should all be removed."
    if st.button("Offensive example"):
        st.session_state["example"] = "You are such a complete idiot, stop talking."
    if st.button("Neutral example"):
        st.session_state["example"] = "I had a wonderful day at the park with my family."

# Main title
st.title("🛡️ SafeNet AI — Hate Speech Detector")
st.markdown("*AI-powered toxicity analysis using NLP & Machine Learning*")
st.divider()

# Input
default_text = st.session_state.get("example", "")
user_text = st.text_area(
    "Enter text to analyse:",
    value=default_text,
    height=140,
    placeholder="Type or paste a social media post here…",
    max_chars=1000,
)
st.caption(str(len(user_text)) + " / 1000 characters")

col1, col2 = st.columns([1, 5])
with col1:
    analyse = st.button("🔍 Analyse", type="primary", use_container_width=True)

if analyse:
    if not user_text.strip():
        st.warning("Please enter some text first.")
    elif model is None:
        st.error("Model not loaded. Run train.py first.")
    else:
        result = predict(user_text, model, vectorizer)
        label  = result["label"]
        emoji  = LABEL_COLOR[label]

        st.divider()
        color_map = {"Hate Speech": "red", "Offensive": "orange", "Neutral": "green"}

        # Result header
        r1, r2, r3 = st.columns(3)
        r1.metric("Classification",  emoji + " " + label)
        r2.metric("Confidence",      str(result["confidence"]) + "%")
        r3.metric("Severity Score",  str(result["severity"]) + " / 100")

        # Progress bars
        st.markdown("**Confidence**")
        st.progress(result["confidence"] / 100)
        st.markdown("**Severity**")
        st.progress(result["severity"] / 100)

        # Class probabilities
        st.markdown("#### Class Probabilities")
        prob_cols = st.columns(3)
        for i, (lid, lname) in enumerate(LABEL_MAP.items()):
            prob = result["all_probs"][lid] if lid < len(result["all_probs"]) else 0.0
            prob_cols[i].metric(lname, str(round(prob * 100, 1)) + "%")

        # Explainability
        if result["top_words"]:
            st.markdown("#### 🔎 Key Contributing Words")
            tags = "  ".join("`" + w + "`" for w in result["top_words"])
            st.markdown(tags)

        # Alert banners
        if label == "Hate Speech":
            st.error("🚨 **Hate Speech Detected** — This content promotes discrimination or hatred.")
        elif label == "Offensive":
            st.warning("⚠️ **Offensive Language** — This content contains offensive or rude language.")
        else:
            st.success("✅ **Neutral Content** — No harmful content detected.")

st.divider()
st.caption("SafeNet AI | NLP Capstone Project | Built with Python, Streamlit & Scikit-learn")
