"""
Model loading and inference for Fake_Job.

Two models are supported:
1. Baseline: TF-IDF + structured features + Logistic Regression
    (artifacts produced by Baseline_Model.ipynb, cell 9, stored locally
    in src/lr_model.pkl, tfidf.pkl, scaler.pkl)
2. DistilBERT: fine-tuned transformer, weights hosted on HuggingFace
    Hub at Yuvraj-Bansal/fake_job-distilbert
"""

import os

import joblib
import numpy as np
import scipy.sparse as sp
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

_ARTIFACT_DIR = os.path.dirname(os.path.abspath(__file__))

LR_MODEL_PATH = os.path.join(_ARTIFACT_DIR, "lr_model.pkl")
TFIDF_PATH = os.path.join(_ARTIFACT_DIR, "tfidf.pkl")
SCALER_PATH = os.path.join(_ARTIFACT_DIR, "scaler.pkl")

HF_MODEL_NAME = "Yuvraj-Bansal/fake_job-distilbert"
DISTILBERT_DEFAULT_THRESHOLD = 0.55  


def load_baseline_model():
    """Load (model, tfidf_vectorizer, scaler) from disk.

    Raises FileNotFoundError with a clear message if any artifact is
    missing, so the Streamlit app can surface a helpful error instead
    of a raw traceback.
    """
    missing = [
        p for p in (LR_MODEL_PATH, TFIDF_PATH, SCALER_PATH) if not os.path.exists(p)
    ]
    if missing:
        names = ", ".join(os.path.basename(p) for p in missing)
        raise FileNotFoundError(
            f"Missing model artifact(s): {names}. "
            f"Make sure lr_model.pkl, tfidf.pkl, and scaler.pkl are present "
            f"in the src/ directory (download from HuggingFace Hub if needed)."
        )

    model = joblib.load(LR_MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, tfidf, scaler


def predict_baseline(cleaned_text: str, struct_arr, model, tfidf, scaler) -> float:
    """Run inference for a single job posting.

    Parameters
    ----------
    cleaned_text : str
        Text already passed through src.preprocess.clean_text().
    struct_arr : array-like, shape (10,)
        Structured features in the exact column order produced by
        src.features.extract_structured_features().
    model, tfidf, scaler : the objects returned by load_baseline_model().

    Returns
    -------
    float
        Predicted probability that the posting is FAKE (class 1).
    """
    # TF-IDF transform (sparse, 1 x n_tfidf_features)
    text_vec = tfidf.transform([cleaned_text])

    # Structured features -> scale -> sparse row vector
    struct_vec = np.asarray(struct_arr, dtype=float).reshape(1, -1)
    struct_scaled = scaler.transform(struct_vec)
    struct_sparse = sp.csr_matrix(struct_scaled)

    # Same column order as training: [tfidf | structured]
    X = sp.hstack([text_vec, struct_sparse], format="csr")

    prob = model.predict_proba(X)[0, 1]
    return float(prob)


# ──────────────────────────────────────────────────────────────
# DistilBERT (HuggingFace Hub)
# ──────────────────────────────────────────────────────────────

def load_distilbert_model():
    """Download (and cache) the fine-tuned DistilBERT model + tokenizer
    from HuggingFace Hub. First call downloads ~250MB; subsequent calls
    on the same machine/container use the local HF cache.
    """
    tokenizer = DistilBertTokenizerFast.from_pretrained(HF_MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(HF_MODEL_NAME)
    model.eval() 
    return model, tokenizer


def load_optimal_threshold(hf_model_name: str = HF_MODEL_NAME,default: float = DISTILBERT_DEFAULT_THRESHOLD) -> float:
    """Download best_threshold.txt from the Hub repo (saved during training's
    threshold sweep) and return it as a float. Falls back to `default`
    (the known training-time value) if the file can't be fetched.
    """
    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(repo_id=hf_model_name, filename="best_threshold.txt")
        with open(path) as f:
            return float(f.read().strip())
    except Exception:
        return default


def build_distilbert_text(title: str, company_profile: str, requirements: str,description: str) -> str:
    return (
        f"Title: {title or ''} [SEP] "
        f"Profile: {company_profile or ''} [SEP] "
        f"Requirements: {requirements or ''} [SEP] "
        f"Description: {description or ''}"
    )


def predict_distilbert(text: str, model, tokenizer) -> float:
    """Returns the predicted probability that the posting is FAKE (class 1).
    Compare against load_optimal_threshold() — NOT a hardcoded 0.5 — to
    convert this probability into a Real/Fake decision.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
    return float(probs[0, 1].item())

