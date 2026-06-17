"""
Structured (non-text) feature engineering for JobGuard.

The column order produced by extract_structured_features() MUST match
the order used when the scaler/model were fit in Baseline_Model.ipynb:

    has_salary, has_company_profile, has_requirements, has_benefits,
    has_logo, telecommuting, desc_length, title_word_count,
    suspicious_title, suspicious_desc
"""

import pandas as pd

# ──────────────────────────────────────────────────────────────
# Suspicious keyword lists (identical to training notebook)
# ──────────────────────────────────────────────────────────────

SUSPICIOUS_TITLE_KEYWORDS = [
    "easy money", "work from home", "no experience needed",
    "no experience required", "unlimited earning", "be your own boss",
    "urgent hiring", "urgently hiring", "home based", "data entry",
    "earn daily", "part time earn", "weekly pay guaranteed", "flexible hours",
    "virtual assistant needed", "start immediately", "immediate start", "no fee"
]

SUSPICIOUS_DESC_KEYWORDS = [
    "whatsapp us", "telegram us", "wire transfer", "western union",
    "money order", "no interview", "guaranteed income",
    "six figure", "financial freedom"
]

STRUCTURED_FEATURE_NAMES = [
    "has_salary", "has_company_profile", "has_requirements", "has_benefits",
    "has_logo", "telecommuting", "desc_length", "title_word_count",
    "suspicious_title", "suspicious_desc"
]


def has_suspicious_keywords(text, keyword_list) -> int:
    """Return 1 if any keyword in keyword_list appears in text (case-insensitive)."""
    if pd.isna(text):
        return 0
    return int(any(kw in str(text).lower() for kw in keyword_list))


def extract_structured_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the 10-column structured feature frame, identical to the
    training notebook's extract_structured_features()."""
    feats = pd.DataFrame(index=df.index)
    feats["has_salary"] = df["salary_range"].notna().astype(int)
    feats["has_company_profile"] = df["company_profile"].notna().astype(int)
    feats["has_requirements"] = df["requirements"].notna().astype(int)
    feats["has_benefits"] = df["benefits"].notna().astype(int)
    feats["has_logo"] = df["has_company_logo"].fillna(0).astype(int)
    feats["telecommuting"] = df["telecommuting"].fillna(0).astype(int)
    feats["desc_length"] = df["description"].fillna("").apply(len)
    feats["title_word_count"] = df["title"].fillna("").apply(lambda x: len(x.split()))
    feats["suspicious_title"] = df["title"].apply(
        lambda x: has_suspicious_keywords(x, SUSPICIOUS_TITLE_KEYWORDS)
    )
    feats["suspicious_desc"] = df["description"].apply(
        lambda x: has_suspicious_keywords(x, SUSPICIOUS_DESC_KEYWORDS)
    )
    return feats
