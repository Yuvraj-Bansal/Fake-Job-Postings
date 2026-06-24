import streamlit as st
import pandas as pd
from src.scraper import scrape_job_posting
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
    page_icon="◆",
    layout="centered"
)

# ──────────────────────────────────────────────────────────────
# Custom styling 
# ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

    .block-container {
        padding-top: 2.5rem;
        max-width: 720px;
    }

    /* Masthead */
    .jg-masthead {
        display: flex;
        align-items: baseline;
        gap: 0.6rem;
        margin-bottom: 0.1rem;
    }
    .jg-mark {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.1rem;
        color: #B5762B;
        font-weight: 600;
    }
    .jg-title {
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: #1E1D1A;
    }
    .jg-subtitle {
        color: #6B6859;
        font-size: 0.92rem;
        margin-top: -0.3rem;
        margin-bottom: 1.6rem;
    }

    /* Section eyebrow labels */
    .jg-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #9A9685;
        margin-bottom: 0.4rem;
        margin-top: 1.8rem;
    }

    hr {
        border-color: #E4E1D8 !important;
        margin: 1.6rem 0 !important;
    }

    /* Verdict readout cards */
    .jg-readout {
        border: 1px solid #E4E1D8;
        border-radius: 6px;
        padding: 1rem 1.1rem;
        background: #FFFFFF;
    }
    .jg-readout-label {
        font-size: 0.78rem;
        color: #6B6859;
        margin-bottom: 0.5rem;
    }
    .jg-readout-verdict {
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .jg-readout-verdict.fake { color: #A33D1F; }
    .jg-readout-verdict.real { color: #3D6B4C; }
    .jg-readout-number {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #57544A;
    }

    /* Risk banner */
    .jg-risk-banner {
        border-left: 3px solid #B5762B;
        padding: 0.7rem 1rem;
        background: #F4EEE3;
        border-radius: 0 4px 4px 0;
        margin: 1rem 0;
    }
    .jg-risk-banner.minimal, .jg-risk-banner.low { border-left-color: #3D6B4C; background: #ECF1ED; }
    .jg-risk-banner.moderate { border-left-color: #B5762B; background: #F4EEE3; }
    .jg-risk-banner.high { border-left-color: #A33D1F; background: #F6E9E4; }

    /* Flag list */
    .jg-flag {
        border-bottom: 1px solid #EFEDE7;
        padding: 0.55rem 0;
        font-size: 0.92rem;
    }
    .jg-flag:last-child { border-bottom: none; }

    .jg-footer {
        color: #9A9685;
        font-size: 0.78rem;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 2.2rem;
    }

    mark {
        background-color: #F6E9E4 !important;
        color: #A33D1F !important;
        padding: 1px 4px;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

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
    model, tokenizer = load_distilbert_model()
    threshold = load_optimal_threshold()
    return model, tokenizer, threshold

lr_model, tfidf_vec, scaler = get_baseline_model()

with st.spinner("Fetching the fine-tuned model."):
    bert_model, bert_tokenizer, bert_threshold = get_distilbert_model()

# ──────────────────────────────────────────────────────────────
# Masthead
# ──────────────────────────────────────────────────────────────

st.markdown("""
<div class="jg-masthead">
    <span class="jg-mark">◆</span>
    <span class="jg-title">Fake Job Detector</span>
</div>
<div class="jg-subtitle">A second opinion on a job posting before you apply.</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Input
# ──────────────────────────────────────────────────────────────

st.markdown('<div class="jg-eyebrow">Start with a link, or type it yourself</div>', unsafe_allow_html=True)

url_input = st.text_input(
    "Job posting URL",
    placeholder="https://www.linkedin.com/jobs/view/...  or any other job board",
    key="job_url",
)

# Session state to hold scraped/typed values
if "scraped" not in st.session_state:
    st.session_state.scraped = {}

if url_input:
    if url_input != st.session_state.get("last_url"):
        with st.spinner("Fetching the posting…"):
            data = scrape_job_posting(url_input)
        st.session_state.scraped = data
        st.session_state.last_url = url_input

        if data["error"]:
            st.warning(f"Couldn't auto-fill: {data['error']}. Fill in the fields below manually.")
        else:
            st.success("Fields pre-filled from the posting. Edit anything that looks off.")

scraped = st.session_state.get("scraped", {})

st.markdown('<div class="jg-eyebrow">The posting</div>', unsafe_allow_html=True)

job_title = st.text_input(
    "Title",
    value=scraped.get("title", ""),
    placeholder="What's the role called?"
)

job_description = st.text_area(
    "Description",
    value=scraped.get("description", ""),
    height=220,
    placeholder="Paste the description as it was written, word for word."
)

with st.expander("Add the company profile or requirements, if listed"):
    st.caption("Optional — but the more of the original posting you paste in, the sharper the read.")
    job_company_profile = st.text_area(
        "Company profile",
        value=scraped.get("company_profile", ""),
        height=90,
        placeholder="The 'about us' blurb, if there was one...",
    )
    job_requirements = st.text_area(
        "Requirements",
        value=scraped.get("requirements", ""),
        height=90,
        placeholder="What they say they're looking for...",
    )

st.markdown('<div class="jg-eyebrow">What else was on the page</div>', unsafe_allow_html=True)
st.caption("Tick what you also saw.")

col1, col2, col3 = st.columns(3)
with col1:
    has_salary = st.checkbox("Salary range", value=scraped.get("has_salary", False))
with col2:
    has_company = st.checkbox("Company profile", value=scraped.get("has_company", False))
with col3:
    has_logo = st.checkbox("Company logo", value=scraped.get("has_logo", False))

col4, col5, col6 = st.columns(3)
with col4:
    has_requirements = st.checkbox("Requirements", value=scraped.get("has_requirements", False))
with col5:
    has_benefits = st.checkbox("Benefits", value=scraped.get("has_benefits", False))
with col6:
    telecommuting = st.checkbox("Remote role", value=scraped.get("telecommuting", False))

# ──────────────────────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────────────────────
analyze = st.button("Analyze Job", type="primary")
if analyze:
    full_text = f"{job_title} {job_description}".strip()

    if len(full_text) < 30:
        st.warning("Add a title and a bit more of the description — there's not enough here to read yet.")
    else:
        # ── Baseline model ──
        suspicious_title = has_suspicious_keywords(job_title, SUSPICIOUS_TITLE_KEYWORDS)
        suspicious_desc = has_suspicious_keywords(job_description, SUSPICIOUS_DESC_KEYWORDS)

        row = pd.DataFrame([{
            "title": job_title,
            "company_profile": "present" if has_company else None,
            "description": job_description,
            "requirements": "present" if has_requirements else None,
            "benefits": "present" if has_benefits else None,
            "salary_range": "present" if has_salary else None,
            "has_company_logo": int(has_logo),
            "telecommuting": int(telecommuting)
        }])

        struct_arr = extract_structured_features(row).iloc[0].values.copy()
        struct_arr = struct_arr.astype(float)
        struct_arr[-2] = suspicious_title
        struct_arr[-1] = suspicious_desc

        cleaned_text = clean_text(full_text)
        baseline_prob = predict_baseline(cleaned_text, struct_arr, lr_model, tfidf_vec, scaler)

        # ── DistilBERT ──
        bert_text = build_distilbert_text(
            title=job_title,
            company_profile=job_company_profile,
            requirements=job_requirements,
            description=job_description
        )
        bert_prob = predict_distilbert(bert_text, bert_model, bert_tokenizer)

        st.markdown('<div class="jg-eyebrow">What the models think</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            is_fake = baseline_prob >= 0.5
            verdict_class = "fake" if is_fake else "real"
            verdict_text = "Reads as fake" if is_fake else "Reads as legitimate"
            confidence = baseline_prob if is_fake else (1 - baseline_prob)
            st.markdown(f"""
            <div class="jg-readout">
                <div class="jg-readout-label">Baseline Model — TF-IDF + Logistic Regression</div>
                <div class="jg-readout-verdict {verdict_class}">{verdict_text}</div>
                <div class="jg-readout-number">{confidence*100:.1f}% confidence · fake-probability {baseline_prob:.3f}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            is_fake_bert = bert_prob >= bert_threshold
            verdict_class_bert = "fake" if is_fake_bert else "real"
            verdict_text_bert = "Reads as fake" if is_fake_bert else "Reads as legitimate"
            confidence_bert = bert_prob if is_fake_bert else (1 - bert_prob)
            st.markdown(f"""
            <div class="jg-readout">
                <div class="jg-readout-label">Fine-Tuned DistilBERT</div>
                <div class="jg-readout-verdict {verdict_class_bert}">{verdict_text_bert}</div>
                <div class="jg-readout-number">{confidence_bert*100:.1f}% confidence · fake-probability {bert_prob:.3f}</div>
            </div>
            """, unsafe_allow_html=True)

        prob = bert_prob
        level, color = risk_level(prob)
        level_class = level.lower().split()[0]

        st.markdown(f"""
        <div class="jg-risk-banner {level_class}">
            <strong>{level}</strong> — based on the DistilBERT read, the stronger of the two models.
        </div>
        """, unsafe_allow_html=True)

        # Red Flags
        flags = find_red_flags(full_text)

        if flags:
            st.write(f"{len(flags)} phrase{'s' if len(flags) != 1 else ''} worth a second look:")
            for flag in flags:
                st.markdown(f'<div class="jg-flag">{flag}</div>', unsafe_allow_html=True)

            highlighted_text, _ = highlight_text(full_text)
            with st.expander("See it marked up in the original text"):
                st.markdown(highlighted_text, unsafe_allow_html=True)
        else:
            st.write("Nothing in the wording itself raised a flag — that alone doesn't clear it, though.")

        st.markdown('<div class="jg-eyebrow">Before you go further</div>', unsafe_allow_html=True)

        if prob >= bert_threshold:
            st.markdown("""
- Don't pay anyone to get a job — not for training, equipment, or a "registration fee."
- Look the company up on LinkedIn — does the recruiter actually work there?
- Check for the role on the company's own careers page, not just this listing.
- Be wary of recruiters who only reach you over WhatsApp.
- Never hand over Aadhaar, PAN, or bank details before an offer is confirmed in writing.
- A salary far above market rate for the role is a flag on its own.
            """)
        else:
            st.markdown("""
- Still worth confirming the company independently before you apply.
- Favor applying through the company's own careers page when you can.
- Cross-check whoever's recruiting you against their LinkedIn profile.
- Keep a copy of every message you exchange with the recruiter.
            """)

# ──────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────

st.markdown("""
<div class="jg-footer">Fake Job Detector — baseline F1 0.82 · DistilBERT F1 0.91, fine-tuned on 17,880 postings</div>
""", unsafe_allow_html=True)
