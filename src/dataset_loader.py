"""
src/dataset_loader.py
Loads Twitter Hate Speech Dataset or generates a realistic sample.
Python 3.14 compatible.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

LABEL_MAP = {0: "Hate Speech", 1: "Offensive Language", 2: "Neutral"}
LABEL_COLORS = {"Hate Speech": "#e74c3c", "Offensive Language": "#f39c12", "Neutral": "#2ecc71"}


def load_dataset(data_dir: str = "./data") -> pd.DataFrame:
    """
    Tries to load dataset.csv from data_dir.
    Falls back to a realistic synthetic sample so the project always runs.
    Returns DataFrame with columns: tweet, label, label_name
    """
    csv_path = Path(data_dir) / "dataset.csv"

    if csv_path.exists():
        logger.info("Loading dataset from %s", csv_path)
        df = pd.read_csv(csv_path)
        df = _normalise_columns(df)
    else:
        logger.info("dataset.csv not found — generating sample dataset")
        df = _generate_sample_dataset()
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        logger.info("Sample dataset saved to %s", csv_path)

    df = _clean_dataset(df)
    logger.info("Dataset loaded: %d rows, labels: %s", len(df), df["label"].value_counts().to_dict())
    return df


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common column names from public datasets to our schema."""
    rename: dict[str, str] = {}

    # Tweet column
    for col in ("tweet", "text", "comment_text", "Text", "Tweet"):
        if col in df.columns:
            rename[col] = "tweet"
            break

    # Label column
    for col in ("class", "label", "Label", "toxic", "category"):
        if col in df.columns and col != "tweet":
            rename[col] = "label"
            break

    df = df.rename(columns=rename)

    # Jigsaw dataset — combine multi-label into single label
    if "label" not in df.columns and "toxic" in df.columns:
        def _jigsaw_label(row):
            if row.get("severe_toxic", 0) == 1 or row.get("threat", 0) == 1:
                return 0  # hate speech
            if row.get("toxic", 0) == 1 or row.get("insult", 0) == 1:
                return 1  # offensive
            return 2      # neutral
        df["label"] = df.apply(_jigsaw_label, axis=1)

    return df


def _clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["tweet"])
    df["tweet"] = df["tweet"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(2).astype(int)
    df["label"] = df["label"].clip(0, 2)
    df["label_name"] = df["label"].map(LABEL_MAP)
    return df.reset_index(drop=True)


def _generate_sample_dataset() -> pd.DataFrame:
    """Generate a realistic 2 000-row sample dataset."""
    rng = np.random.default_rng(42)

    hate_speech_samples = [
        "I hate all people from that country they should leave",
        "Those immigrants are ruining everything get out",
        "All members of that group are dangerous criminals",
        "We need to eliminate these people from our society",
        "That religious group is a plague on our nation",
        "Kill all the vermin who support that ideology",
        "These subhumans do not deserve rights",
        "Exterminate the enemy within our borders",
        "This race is inferior and should not exist",
        "Down with those disgusting people and their culture",
    ] * 60  # ~600 samples

    offensive_samples = [
        "That was a stupid idea what were you thinking",
        "You are such an idiot for believing that garbage",
        "This movie is absolute trash and so are the actors",
        "Shut up nobody cares about your opinion",
        "Go back to where you came from loser",
        "You dumb fool this is the worst thing I have ever seen",
        "These politicians are all corrupt liars",
        "What kind of moron would do something like that",
        "This whole system is broken and the people running it are fools",
        "I cannot believe how utterly useless some people are",
    ] * 80  # ~800 samples

    neutral_samples = [
        "The weather today is absolutely beautiful I love sunny days",
        "Just finished reading a great book would highly recommend it",
        "Had an amazing dinner with family last night so grateful",
        "Looking forward to the weekend trip we have planned",
        "The new park opening downtown looks really impressive",
        "I learned a new recipe today and it turned out delicious",
        "Great game last night the team played really well",
        "Just watched an interesting documentary about space exploration",
        "The library has a wonderful collection of science fiction novels",
        "Happy to share that my project got selected for the science fair",
    ] * 60  # ~600 samples

    tweets = hate_speech_samples[:600] + offensive_samples[:800] + neutral_samples[:600]
    labels = [0] * 600 + [1] * 800 + [2] * 600

    # Shuffle
    idx = rng.permutation(len(tweets))
    tweets = [tweets[i] for i in idx]
    labels = [labels[i] for i in idx]

    return pd.DataFrame({"tweet": tweets, "label": labels})
