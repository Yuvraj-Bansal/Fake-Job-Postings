"""
Model loading and inference for JobGuard's baseline
(TF-IDF + structured features + Logistic Regression).

Artifacts expected (produced by Baseline_Model.ipynb, cell 9):
    src/lr_model.pkl   - trained sklearn LogisticRegression
    src/tfidf.pkl      - fitted TfidfVectorizer
    src/scaler.pkl     - fitted MaxAbsScaler (for structured features)
"""

import os

import joblib
import numpy as np
import scipy.sparse as sp

_ARTIFACT_DIR = os.path.dirname(os.path.abspath(__file__))

LR_MODEL_PATH = os.path.join(_ARTIFACT_DIR, "lr_model.pkl")
TFIDF_PATH = os.path.join(_ARTIFACT_DIR, "tfidf.pkl")
SCALER_PATH = os.path.join(_ARTIFACT_DIR, "scaler.pkl")


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
