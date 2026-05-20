"""
Hate Speech Detection — Training Pipeline
SafeNet AI | Python 3.14 Compatible
=====================================================
Trains and compares: Logistic Regression, Naive Bayes, Random Forest, SVM
Saves the best model + vectorizer for deployment.
"""
from __future__ import annotations

import os
import re
import json
import pickle
import logging
import warnings
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
for d in (DATA_DIR, MODEL_DIR, OUTPUT_DIR):
    d.mkdir(exist_ok=True)

LABEL_MAP = {0: "Hate Speech", 1: "Offensive", 2: "Neutral"}
LABEL_COLORS = {"Hate Speech": "#e74c3c", "Offensive": "#f39c12", "Neutral": "#2ecc71"}


# ── 1. NLTK bootstrap ──────────────────────────────────────────────────────
def download_nltk_resources() -> None:
    for resource in ("punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"):
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass


# ── 2. DATASET ─────────────────────────────────────────────────────────────
def load_or_generate_dataset() -> pd.DataFrame:
    """
    Tries to load data/dataset.csv.
    Falls back to a realistic synthetic dataset so the project runs
    on any machine without a Kaggle API key.
    """
    csv_path = DATA_DIR / "dataset.csv"
    if csv_path.exists():
        logger.info("Loading dataset from %s", csv_path)
        df = pd.read_csv(csv_path)
        # Normalise common column names
        col_map = {}
        for c in df.columns:
            if c.lower() in ("tweet", "text", "comment", "sentence"):
                col_map[c] = "tweet"
            if c.lower() in ("class", "label", "category"):
                col_map[c] = "label"
        df = df.rename(columns=col_map)
        if "tweet" not in df.columns or "label" not in df.columns:
            raise ValueError("Dataset must have 'tweet' and 'label' columns.")
        return df

    logger.warning("dataset.csv not found — generating synthetic demo dataset.")
    return _generate_synthetic_dataset()


def _generate_synthetic_dataset() -> pd.DataFrame:
    """Creates a balanced 900-sample synthetic dataset for demonstration."""
    rng = np.random.default_rng(42)

    hate_samples = [
        "I hate all people who are different from us",
        "They should all be removed from this country",
        "These people are a plague on society",
        "Everyone knows that group X is inferior",
        "We need to eliminate those disgusting creatures",
        "They don't deserve to live among us",
        "Those animals don't belong here get them out",
        "Complete trash this entire group needs to disappear",
        "All members of that religion are terrorists",
        "Burn their places of worship to the ground",
        "They should be banned from entering our nation",
        "Disgusting subhumans invading our land",
        "Those people are genetically inferior to us",
        "We must stand against the invasion of our culture",
        "Filthy criminals from that group everywhere",
    ] * 20

    offensive_samples = [
        "That was a really stupid thing to say",
        "You are such an idiot for thinking that",
        "What a complete moron honestly",
        "This is garbage content stop posting",
        "Shut up nobody cares what you think",
        "You are pathetically wrong about everything",
        "Dumbest take I have ever seen online",
        "You clearly have no brain cells left",
        "Stop being so incredibly annoying all the time",
        "That is a terrible and dumb opinion",
        "You are wasting everyone's time with this",
        "Absolute nonsense from beginning to end",
        "Seriously the most ignorant thing I have read",
        "You have no idea what you are talking about",
        "This post is embarrassingly bad and wrong",
    ] * 20

    neutral_samples = [
        "I had a great time at the park today",
        "The weather is really nice this morning",
        "Just finished reading an amazing book",
        "Looking forward to the weekend with family",
        "The new restaurant downtown is excellent",
        "Today was a productive day at work",
        "I love learning new things every day",
        "The movie last night was quite entertaining",
        "Going for a walk helps clear my mind",
        "Cooking a new recipe for dinner tonight",
        "The sunrise this morning was beautiful",
        "Had a wonderful coffee chat with a friend",
        "Finally finished my project after weeks of work",
        "The garden is blooming beautifully this spring",
        "Listening to great music while working",
    ] * 20

    tweets = hate_samples + offensive_samples + neutral_samples
    labels = [0] * len(hate_samples) + [1] * len(offensive_samples) + [2] * len(neutral_samples)

    df = pd.DataFrame({"tweet": tweets, "label": labels})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(DATA_DIR / "dataset.csv", index=False)
    logger.info("Synthetic dataset saved (%d rows)", len(df))
    return df


# ── 3. PREPROCESSING ───────────────────────────────────────────────────────
class TextPreprocessor:
    """Smart preprocessing pipeline: clean → tokenise → lemmatise."""

    def __init__(self) -> None:
        download_nltk_resources()
        self.lemmatiser = WordNetLemmatizer()
        try:
            self._stops = set(stopwords.words("english"))
        except Exception:
            self._stops = set()

    def clean(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", " ", text)          # URLs
        text = re.sub(r"@\w+", " ", text)                     # mentions
        text = re.sub(r"#(\w+)", r"\1", text)                 # hashtags → word
        text = re.sub(r"[^a-zA-Z\s]", " ", text)             # keep only letters
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenise_and_lemmatise(self, text: str) -> str:
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()
        tokens = [
            self.lemmatiser.lemmatize(t)
            for t in tokens
            if t not in self._stops and len(t) > 2
        ]
        return " ".join(tokens)

    def process(self, text: str) -> str:
        return self.tokenise_and_lemmatise(self.clean(text))

    def process_series(self, series: pd.Series) -> pd.Series:
        logger.info("Preprocessing %d texts…", len(series))
        return series.fillna("").apply(self.process)


# ── 4. FEATURE ENGINEERING ─────────────────────────────────────────────────
def build_features(
    train_texts: pd.Series,
    test_texts:  pd.Series,
    max_features: int = 10_000,
) -> tuple[object, object, object]:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test  = vectorizer.transform(test_texts)
    return vectorizer, X_train, X_test


# ── 5. MODELS ──────────────────────────────────────────────────────────────
def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=1.0),
        "Naive Bayes":         MultinomialNB(alpha=0.5),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "SVM (LinearSVC)":     LinearSVC(max_iter=2000, random_state=42, C=0.5),
    }


def train_and_evaluate(
    models: dict,
    X_train: object,
    X_test:  object,
    y_train: pd.Series,
    y_test:  pd.Series,
) -> tuple[dict, dict, str]:
    results: dict = {}
    reports: dict = {}
    best_name = ""
    best_f1   = 0.0

    for name, clf in models.items():
        logger.info("Training %s…", name)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        acc  = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted", zero_division=0)
        rec  = recall_score(y_test, preds, average="weighted", zero_division=0)
        f1   = f1_score(y_test, preds, average="weighted", zero_division=0)

        results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1}
        reports[name] = classification_report(
            y_test, preds,
            target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP)],
            zero_division=0,
        )
        logger.info("  %s → Acc=%.4f  F1=%.4f", name, acc, f1)

        if f1 > best_f1:
            best_f1, best_name = f1, name

    return results, reports, best_name


# ── 6. VISUALISATIONS ──────────────────────────────────────────────────────
def plot_class_distribution(df: pd.DataFrame) -> None:
    counts = df["label"].value_counts().sort_index()
    labels = [LABEL_MAP[i] for i in counts.index]
    colors = [LABEL_COLORS[l] for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Class Distribution", fontsize=15, fontweight="bold")

    axes[0].bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.2)
    axes[0].set_title("Bar Chart")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 5, str(v), ha="center", fontweight="bold")

    axes[1].pie(counts.values, labels=labels, colors=colors,
                autopct="%1.1f%%", startangle=90, pctdistance=0.8)
    axes[1].set_title("Pie Chart")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved class_distribution.png")


def plot_confusion_matrix(clf, X_test, y_test, model_name: str) -> None:
    preds = clf.predict(X_test)
    cm    = confusion_matrix(y_test, preds)
    names = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=names, yticklabels=names, ax=ax)
    ax.set_title(model_name + " — Confusion Matrix", fontsize=13, fontweight="bold")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(OUTPUT_DIR / ("cm_" + safe_name + ".png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_model_comparison(results: dict) -> None:
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    model_names = list(results.keys())
    x = np.arange(len(model_names))
    width = 0.2
    colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [results[m][metric] for m in model_names]
        bars = ax.bar(x + i * width, vals, width, label=metric, color=color, alpha=0.87)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    str(round(bar.get_height(), 3)),
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — All Metrics", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved model_comparison.png")


def plot_wordcloud(df: pd.DataFrame) -> None:
    try:
        from wordcloud import WordCloud
    except ImportError:
        logger.warning("wordcloud not installed — skipping word cloud.")
        return

    for label_id, label_name in LABEL_MAP.items():
        texts = " ".join(df[df["label"] == label_id]["clean_text"].fillna(""))
        if not texts.strip():
            continue
        wc = WordCloud(
            width=800, height=400,
            background_color="white",
            colormap="Reds" if label_name == "Hate Speech" else
                     "Oranges" if label_name == "Offensive" else "Greens",
            max_words=100,
        ).generate(texts)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(label_name + " — Word Cloud", fontsize=14, fontweight="bold")
        plt.tight_layout()
        safe = label_name.lower().replace(" ", "_")
        plt.savefig(OUTPUT_DIR / ("wordcloud_" + safe + ".png"), dpi=150, bbox_inches="tight")
        plt.close()
    logger.info("Saved word clouds.")


# ── 7. SAVE ARTEFACTS ──────────────────────────────────────────────────────
def save_artefacts(
    best_clf,
    vectorizer,
    best_name: str,
    results: dict,
    label_encoder: LabelEncoder | None = None,
) -> None:
    with open(MODEL_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(best_clf, f)
    with open(MODEL_DIR / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    meta = {
        "best_model": best_name,
        "label_map":  LABEL_MAP,
        "results":    {m: {k: round(v, 4) for k, v in r.items()} for m, r in results.items()},
    }
    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("Artefacts saved to models/")


# ── 8. MAIN ────────────────────────────────────────────────────────────────
def main() -> None:
    logger.info("=== SafeNet AI — Training Pipeline ===")

    # Load
    df = load_or_generate_dataset()
    logger.info("Dataset shape: %s", df.shape)
    logger.info("Label distribution:\n%s", df["label"].value_counts().to_string())

    # Preprocess
    preprocessor = TextPreprocessor()
    df["clean_text"] = preprocessor.process_series(df["tweet"])

    # Visualise data
    plot_class_distribution(df)
    plot_wordcloud(df)

    # Split
    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        df["clean_text"], df["label"],
        test_size=0.20, random_state=42, stratify=df["label"],
    )

    # Features
    vectorizer, X_train, X_test = build_features(X_train_txt, X_test_txt)

    # Train & evaluate
    models  = get_models()
    results, reports, best_name = train_and_evaluate(models, X_train, X_test, y_train, y_test)

    # Print results table
    res_df = pd.DataFrame(results).T.round(4)
    logger.info("\nModel Comparison:\n%s", res_df.to_string())

    # Save reports
    with open(OUTPUT_DIR / "classification_reports.txt", "w", encoding="utf-8") as f:
        for name, report in reports.items():
            f.write("=" * 60 + "\n")
            f.write(name + "\n")
            f.write("=" * 60 + "\n")
            f.write(report + "\n\n")

    # Save results CSV
    res_df.to_csv(OUTPUT_DIR / "model_results.csv")

    # Confusion matrices + comparison chart
    for name, clf in models.items():
        plot_confusion_matrix(clf, X_test, y_test, name)
    plot_model_comparison(results)

    # Save best model
    best_clf = models[best_name]
    save_artefacts(best_clf, vectorizer, best_name, results)

    logger.info("Best model: %s", best_name)
    logger.info("=== Training complete. Check outputs/ and models/ ===")


if __name__ == "__main__":
    main()
