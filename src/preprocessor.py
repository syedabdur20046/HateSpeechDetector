"""
src/preprocessor.py
Smart Text Preprocessing Pipeline
Python 3.14 compatible — built-in generics only, no deprecated typing
"""
from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)

_NLTK_READY = False

def _ensure_nltk() -> None:
    global _NLTK_READY
    if _NLTK_READY:
        return
    import nltk
    for res in ("stopwords", "wordnet", "punkt", "omw-1.4", "punkt_tab"):
        try:
            nltk.download(res, quiet=True)
        except Exception:
            pass
    _NLTK_READY = True


class TextPreprocessor:
    """
    Pipeline: lowercase → URL strip → mention/hashtag → slang expand
              → special chars → stopwords → lemmatize
    """

    SLANG_MAP: dict[str, str] = {
        "u":    "you",      "ur":   "your",     "r":    "are",
        "4":    "for",      "gr8":  "great",    "lol":  "laughing",
        "omg":  "oh my god","tbh":  "to be honest",
        "imo":  "in my opinion",                "smh":  "shaking my head",
        "ngl":  "not gonna lie",               "wtf":  "what the heck",
    }

    def __init__(self, use_lemmatizer: bool = True, remove_stopwords: bool = True) -> None:
        _ensure_nltk()
        self.use_lemmatizer   = use_lemmatizer
        self.remove_stopwords = remove_stopwords
        self._lemmatizer      = None
        self._stop_words: set[str] = set()
        self._init_nlp()

    def _init_nlp(self) -> None:
        try:
            from nltk.stem   import WordNetLemmatizer
            from nltk.corpus import stopwords
            self._lemmatizer = WordNetLemmatizer()
            sw = set(stopwords.words("english"))
            # Keep negations — critical for toxicity detection
            sw -= {"no", "not", "nor", "never", "neither", "none"}
            self._stop_words = sw
        except Exception as exc:
            logger.warning("NLTK init issue: %s", exc)

    def clean(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = re.sub(r"@\w+", " ", text)
        text = re.sub(r"#(\w+)", r" \1 ", text)
        tokens = text.split()
        tokens = [self.SLANG_MAP.get(t, t) for t in tokens]
        text = " ".join(tokens)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        if self.remove_stopwords and self._stop_words:
            text = " ".join(w for w in text.split() if w not in self._stop_words)
        if self.use_lemmatizer and self._lemmatizer:
            text = " ".join(self._lemmatizer.lemmatize(w) for w in text.split())
        return text.strip()

    def clean_batch(self, texts: list[str]) -> list[str]:
        return [self.clean(t) for t in texts]
