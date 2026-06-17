import streamlit as st
import pandas as pd

from src.preprocess import clean_text
from src.features import (
    extract_structured_features,
    has_suspicious_keywords,
    SUSPICIOUS_TITLE_KEYWORDS,
    SUSPICIOUS_DESC_KEYWORDS
)
from src.model import load_baseline_model, predict_baseline
from src.explainer import (
    find_red_flags,
    highlight_text,
    risk_level
)

# ──────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="JobGuard",
    page_icon="🛡️",
    layout="centered"
)

# ──────────────────────────────────────────────────────────────
# Load Model
# ──────────────────────────────────────────────────────────────

@st.cache_resource
def get_model():
    try:
        model, tfidf, scaler = load_baseline_model()
        return model, tfidf, scaler
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

lr_model, tfidf_vec, scaler = get_model()

# ──────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────

st.title("🛡️ JobGuard")
st.caption("Detect Fake Job Postings using Machine Learning")

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
        # Detect suspicious keywords
        suspicious_title = has_suspicious_keywords(
            job_title,
            SUSPICIOUS_TITLE_KEYWORDS
        )

        suspicious_desc = has_suspicious_keywords(
            job_description,
            SUSPICIOUS_DESC_KEYWORDS
        )

        # Build dataframe matching training schema
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

        # Generate structured features
        struct_arr = extract_structured_features(
            row
        ).iloc[0].values.copy()  # .copy() avoids "read-only array" errors
        struct_arr = struct_arr.astype(float)

        # Overwrite auto-generated suspicious flags
        struct_arr[-2] = suspicious_title
        struct_arr[-1] = suspicious_desc

        # Clean text
        cleaned_text = clean_text(full_text)

        # Predict
        prob = predict_baseline(
            cleaned_text,
            struct_arr,
            lr_model,
            tfidf_vec,
            scaler
        )

        # Risk level
        level, color = risk_level(prob)

        st.divider()
        st.subheader("🔎 Analysis Result")

        if prob >= 0.5:
            st.error(
                f"⚠️ Likely FAKE Job Posting ({prob*100:.1f}% confidence)"
            )
        else:
            st.success(
                f"✅ Appears Legitimate ({(1-prob)*100:.1f}% confidence)"
            )

        st.progress(
            float(prob),
            text=f"Fake Probability: {prob*100:.1f}%"
        )

        st.markdown(
            f"### Risk Level: :{color}[{level}]"
        )

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
                unsafe_allow_html=True  # Changed to True so your highlight html spans actually render color
            )
        else:
            st.info(
                "✅ No obvious scam phrases detected."
            )

        st.divider()
        st.subheader("💡 Safety Tips")

        if prob >= 0.5:
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
    "JobGuard • TF-IDF + Logistic Regression + Structured Features"
)
