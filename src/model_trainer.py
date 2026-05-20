"""
src/model_trainer.py
Trains and evaluates Logistic Regression, Naive Bayes, Random Forest, SVM.
Python 3.14 compatible.
"""
from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model     import LogisticRegression
from sklearn.naive_bayes      import MultinomialNB
from sklearn.ensemble         import RandomForestClassifier
from sklearn.svm              import LinearSVC
from sklearn.model_selection  import StratifiedKFold, cross_val_score
from sklearn.metrics          import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
from sklearn.calibration      import CalibratedClassifierCV

logger = logging.getLogger(__name__)

LABEL_NAMES = ["Hate Speech", "Offensive Language", "Neutral"]


def _make_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs", multi_class="multinomial", random_state=42
        ),
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
        ),
        "SVM": CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=42)),
    }


class ModelTrainer:
    """Train, compare, and persist multiple classifiers."""

    def __init__(self) -> None:
        self.models          = _make_models()
        self.results: dict   = {}
        self.best_model_name = ""
        self.best_model      = None

    def train_all(self, X_train, y_train, X_test, y_test) -> pd.DataFrame:
        """Train all models, return comparison DataFrame."""
        rows: list[dict] = []
        for name, model in self.models.items():
            logger.info("Training: %s", name)
            t0 = time.time()
            model.fit(X_train, y_train)
            train_time = round(time.time() - t0, 2)

            y_pred = model.predict(X_test)
            metrics = {
                "Model":     name,
                "Accuracy":  round(accuracy_score(y_test, y_pred),                     4),
                "Precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
                "Recall":    round(recall_score(y_test, y_pred, average="weighted",    zero_division=0), 4),
                "F1-Score":  round(f1_score(y_test, y_pred, average="weighted",        zero_division=0), 4),
                "Train Time (s)": train_time,
            }
            self.results[name] = {
                "metrics":        metrics,
                "y_pred":         y_pred,
                "conf_matrix":    confusion_matrix(y_test, y_pred),
                "class_report":   classification_report(y_test, y_pred, target_names=LABEL_NAMES, zero_division=0),
            }
            rows.append(metrics)
            logger.info("  Accuracy=%.4f  F1=%.4f  Time=%ss", metrics["Accuracy"], metrics["F1-Score"], train_time)

        comparison_df = pd.DataFrame(rows).sort_values("F1-Score", ascending=False).reset_index(drop=True)

        # Store best model
        self.best_model_name = comparison_df.iloc[0]["Model"]
        self.best_model      = self.models[self.best_model_name]
        logger.info("Best model: %s (F1=%.4f)", self.best_model_name, comparison_df.iloc[0]["F1-Score"])
        return comparison_df

    def predict(self, X, model_name: str | None = None):
        """Predict labels using the specified (or best) model."""
        model = self.models.get(model_name, self.best_model)
        if model is None:
            raise RuntimeError("No trained model available.")
        return model.predict(X)

    def predict_proba(self, X, model_name: str | None = None) -> np.ndarray:
        """Return class probabilities (requires predict_proba support)."""
        model = self.models.get(model_name, self.best_model)
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        # Fallback: one-hot from hard prediction
        preds = model.predict(X)
        proba = np.zeros((len(preds), 3))
        for i, p in enumerate(preds):
            proba[i, p] = 1.0
        return proba

    def save_best(self, models_dir: str) -> None:
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        path = Path(models_dir) / "best_model.pkl"
        with open(path, "wb") as f:
            pickle.dump({"name": self.best_model_name, "model": self.best_model}, f)
        logger.info("Best model (%s) saved to %s", self.best_model_name, path)

    def load_best(self, models_dir: str) -> None:
        path = Path(models_dir) / "best_model.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.best_model_name = data["name"]
        self.best_model      = data["model"]
        self.models[self.best_model_name] = self.best_model
        logger.info("Loaded model: %s", self.best_model_name)
