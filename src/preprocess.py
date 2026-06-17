"""
Text preprocessing utilities for JobGuard.

IMPORTANT: clean_text() here must stay byte-for-byte identical to the
clean_text() used in Baseline_Model.ipynb during training. The TF-IDF
vectorizer was fit on text produced by this exact function, so any
drift here will silently degrade prediction quality at inference time.
"""

import re

import nltk
import pandas as pd

# Ensure stopwords are available (no-op if already downloaded)
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords  # noqa: E402

STOP_WORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Lowercase, strip HTML/URLs/punctuation, remove stopwords and
    short tokens. Mirrors the training notebook exactly."""
    if pd.isna(text) or text is None:
        return ""
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return " ".join(tokens)


def build_combined_text(df: pd.DataFrame) -> pd.Series:
    """Combine title + company_profile + description + requirements +
    benefits into one cleaned text blob per row (training-time helper,
    kept here for parity / potential batch re-use)."""
    cols = ["title", "company_profile", "description", "requirements", "benefits"]
    combined = df[cols].fillna("").agg(" ".join, axis=1)
    return combined.apply(clean_text)
