import streamlit as st
import pandas as pd

from src.preprocess import clean_text
from src.features import (
    extract_structured_features,
    has_suspicious_keywords,
    SUSPICIOUS_TITLE_KEYWORDS,
    SUSPICIOUS_DESC_KEYWORDS
)
from src.model import (
    load_baseline_model, predict_baseline,
    load_distilbert_model, predict_distilbert,
    load_optimal_threshold, build_distilbert_text
)
from src.explainer import (
    find_red_flags,
    highlight_text,
    risk_level
)

# ──────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="🛡️",
    layout="centered"
)

# ──────────────────────────────────────────────────────────────
# Load Models
# ──────────────────────────────────────────────────────────────

@st.cache_resource
def get_baseline_model():
    try:
        model, tfidf, scaler = load_baseline_model()
        return model, tfidf, scaler
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

@st.cache_resource
def get_distilbert_model():
    # Downloads ~250MB from HuggingFace Hub on first run, then caches
    # for the rest of the app's lifetime (not re-downloaded per click).
    model, tokenizer = load_distilbert_model()
    threshold = load_optimal_threshold()
    return model, tokenizer, threshold

lr_model, tfidf_vec, scaler = get_baseline_model()

with st.spinner("Loading DistilBERT model from HuggingFace Hub ..."):
    bert_model, bert_tokenizer, bert_threshold = get_distilbert_model()

# ──────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────

st.title("🛡️ Fake Job Detector")
st.caption("Detect Fake Job Postings")

st.divider()

st.subheader("📋 Job Information")

job_title = st.text_input(
    "Job Title",
    placeholder="e.g. Data Entry Operator"
)

job_description = st.text_area(
    "Job Description",
    height=250,
    placeholder="Paste the full job description..."
)

with st.expander("➕ Optional: Company Profile & Requirements (improves accuracy)"):
    job_company_profile = st.text_area(
        "Company Profile",
        height=100,
        placeholder="Paste the company's 'About Us' text, if listed..."
    )
    job_requirements = st.text_area(
        "Requirements",
        height=100,
        placeholder="Paste the listed job requirements, if any..."
    )

st.divider()

st.subheader("📊 Additional Information")

col1, col2, col3 = st.columns(3)

with col1:
    has_salary = st.checkbox("💰 Salary Mentioned")

with col2:
    has_company = st.checkbox("🏢 Company Profile")

with col3:
    has_logo = st.checkbox("🖼️ Company Logo")

col4, col5, col6 = st.columns(3)

with col4:
    has_requirements = st.checkbox("📝 Requirements Listed")

with col5:
    has_benefits = st.checkbox("🎁 Benefits Listed")

with col6:
    telecommuting = st.checkbox("🏠 Remote Job")

st.divider()

analyze = st.button(
    "🔍 Analyze Job Posting",
    type="primary",
    use_container_width=True
)

# ──────────────────────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────────────────────

if analyze:
    full_text = f"{job_title} {job_description}".strip()

    if len(full_text) < 30:
        st.warning(
            "Please provide a job title and a more detailed description."
        )
    else:
        # ── Baseline model (TF-IDF + Logistic Regression) ──

        suspicious_title = has_suspicious_keywords(
            job_title,
            SUSPICIOUS_TITLE_KEYWORDS
        )
        suspicious_desc = has_suspicious_keywords(
            job_description,
            SUSPICIOUS_DESC_KEYWORDS
        )

        row = pd.DataFrame([{
            "title": job_title,
            "company_profile": (
                "present" if has_company else None
            ),
            "description": job_description,
            "requirements": (
                "present" if has_requirements else None
            ),
            "benefits": (
                "present" if has_benefits else None
            ),
            "salary_range": (
                "present" if has_salary else None
            ),
            "has_company_logo": int(has_logo),
            "telecommuting": int(telecommuting)
        }])

        struct_arr = extract_structured_features(
            row
        ).iloc[0].values.copy()  # .copy() avoids "read-only array" errors
        struct_arr = struct_arr.astype(float)

        struct_arr[-2] = suspicious_title
        struct_arr[-1] = suspicious_desc

        cleaned_text = clean_text(full_text)

        baseline_prob = predict_baseline(
            cleaned_text,
            struct_arr,
            lr_model,
            tfidf_vec,
            scaler
        )

        # ── DistilBERT model ──
        bert_text = build_distilbert_text(
            title=job_title,
            company_profile=job_company_profile,
            requirements=job_requirements,
            description=job_description
        )
        bert_prob = predict_distilbert(bert_text, bert_model, bert_tokenizer)

        st.divider()
        st.subheader("🔎 Analysis Result")

        # Side-by-side model comparison
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Baseline (TF-IDF + LR)**")
            if baseline_prob >= 0.5:
                st.error(f"⚠️ Likely FAKE ({baseline_prob*100:.1f}%)")
            else:
                st.success(f"✅ Likely Real ({(1-baseline_prob)*100:.1f}%)")
            st.progress(float(baseline_prob), text=f"Fake prob: {baseline_prob*100:.1f}%")

        with col_b:
            st.markdown("**DistilBERT (fine-tuned)**")
            if bert_prob >= bert_threshold:
                st.error(f"⚠️ Likely FAKE ({bert_prob*100:.1f}%)")
            else:
                st.success(f"✅ Likely Real ({(1-bert_prob)*100:.1f}%)")
            st.progress(float(bert_prob), text=f"Fake prob: {bert_prob*100:.1f}%")

        st.caption(
            f"DistilBERT uses an optimized decision threshold of {bert_threshold:.2f} "
            f"(tuned for best F1 during training), not the default 0.5."
        )

        prob = bert_prob
        level, color = risk_level(prob)

        st.divider()
        st.markdown(f"### Overall Risk Level: :{color}[{level}]")
        st.caption("Based on the DistilBERT model's prediction (higher validation F1 than baseline).")

        st.divider()

        # Red Flags
        flags = find_red_flags(full_text)

        if flags:
            st.subheader(
                f"🚩 {len(flags)} Red Flag(s) Detected"
            )

            for flag in flags:
                st.warning(flag)

            highlighted_text, _ = highlight_text(
                full_text
            )

            st.subheader("📄 Highlighted Text")
            st.markdown(
                highlighted_text,
                unsafe_allow_html=True  
            )
        else:
            st.info(
                "✅ No obvious scam phrases detected."
            )

        st.divider()
        st.subheader("💡 Safety Tips")

        if prob >= bert_threshold:
            st.markdown("""
* Never pay money to get a job.
* Verify the company on LinkedIn.
* Check the company's official website.
* Avoid WhatsApp-only recruiters.
* Never share Aadhaar/PAN before verification.
* Be cautious of unrealistic salaries.
            """)
        else:
            st.markdown("""
* Verify company details before applying.
* Prefer official company career portals.
* Cross-check recruiter information on LinkedIn.
* Keep records of all communication.
            """)

# ──────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "TF-IDF + Logistic Regression baseline (F1 0.82) • "
    "DistilBERT fine-tuned (F1 0.91) • Structured Features"
)
