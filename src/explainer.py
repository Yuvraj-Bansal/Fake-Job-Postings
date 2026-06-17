"""
Human-readable explanation layer for JobGuard.

This module turns raw text + suspicious keyword matches into:
- a list of plain-English "red flags" the user can read
- an HTML-highlighted version of the job text (suspicious phrases wrapped
    in colored <mark> spans)
- a risk_level() helper that maps a probability to a label + Streamlit
    color name (used as st.markdown(f":{color}[{level}]"))

None of this changes model inputs/outputs — it's purely a presentation
layer built on top of the same keyword lists used for the
suspicious_title / suspicious_desc structured features.
"""

import html
import re

from src.features import SUSPICIOUS_DESC_KEYWORDS, SUSPICIOUS_TITLE_KEYWORDS

# ──────────────────────────────────────────────────────────────
# Red flag phrase -> human-readable explanation
# ──────────────────────────────────────────────────────────────

_ALL_FLAG_PHRASES = sorted(
    set(SUSPICIOUS_TITLE_KEYWORDS) | set(SUSPICIOUS_DESC_KEYWORDS),
    key=len,
    reverse=True,  # match longer phrases first so substrings don't shadow them
)

_FLAG_EXPLANATIONS = {
    "easy money": "Claims of 'easy money' are a classic scam lure.",
    "work from home": "Vague 'work from home' framing with no real job details.",
    "no experience needed": "No experience required — often used to lower applicants' guard.",
    "no experience required": "No experience required — often used to lower applicants' guard.",
    "unlimited earning": "Unrealistic 'unlimited earning' promises.",
    "be your own boss": "Vague self-employment pitch typical of pyramid/MLM scams.",
    "urgent hiring": "Artificial urgency pressures applicants to skip due diligence.",
    "urgently hiring": "Artificial urgency pressures applicants to skip due diligence.",
    "home based": "Generic 'home based' work claim with little verifiable detail.",
    "data entry": "Data entry roles are frequently used in fake job scams.",
    "earn daily": "Daily payout promises are a common scam pattern.",
    "part time earn": "Vague part-time earning claims, often unverifiable.",
    "weekly pay guaranteed": "Guaranteed pay claims are a red flag for fraud.",
    "flexible hours": "Vague 'flexible hours' language with no role specifics.",
    "virtual assistant needed": "Generic virtual assistant postings are commonly faked.",
    "start immediately": "Pressure to start immediately, skipping normal hiring steps.",
    "immediate start": "Pressure to start immediately, skipping normal hiring steps.",
    "no fee": "Explicitly mentioning 'no fee' can be a tell that fees are otherwise expected.",
    "whatsapp us": "Recruitment via WhatsApp instead of official channels is a major red flag.",
    "telegram us": "Recruitment via Telegram instead of official channels is a major red flag.",
    "wire transfer": "Requests involving wire transfers are strongly associated with scams.",
    "western union": "Western Union is frequently used in job-scam payment fraud.",
    "money order": "Money order requests are a classic advance-fee scam tactic.",
    "no interview": "Hiring with no interview process is highly unusual and risky.",
    "guaranteed income": "'Guaranteed income' claims are unrealistic for legitimate roles.",
    "six figure": "Unrealistic six-figure income promises for unspecified roles.",
    "financial freedom": "'Financial freedom' language is common in MLM/scam pitches.",
}


def find_red_flags(text: str):
    """Scan text for known suspicious phrases and return a list of
    human-readable warning strings (one per unique phrase found)."""
    if not text:
        return []

    lowered = text.lower()
    found = []
    seen = set()

    for phrase in _ALL_FLAG_PHRASES:
        if phrase in lowered and phrase not in seen:
            seen.add(phrase)
            explanation = _FLAG_EXPLANATIONS.get(
                phrase, f"Suspicious phrase detected: '{phrase}'."
            )
            found.append(f"**\"{phrase}\"** — {explanation}")

    return found


def highlight_text(text: str):
    """Return (html_string, list_of_matched_phrases) where every
    suspicious phrase in `text` is wrapped in a colored <mark> span.
    Safe against HTML injection: the base text is escaped first, then
    marks are inserted around matched (also escaped) phrases.
    """
    if not text:
        return "", []

    escaped = html.escape(text)
    matched = []

    for phrase in _ALL_FLAG_PHRASES:
        escaped_phrase = html.escape(phrase)
        pattern = re.compile(re.escape(escaped_phrase), re.IGNORECASE)

        def _wrap(m, _phrase=phrase):
            matched.append(_phrase)
            return (
                '<mark style="background-color:#ffcdd2;color:#b71c1c;'
                f'padding:2px 4px;border-radius:3px;">{m.group(0)}</mark>'
            )

        escaped = pattern.sub(_wrap, escaped)

    # Preserve line breaks for Streamlit markdown rendering
    html_out = escaped.replace("\n", "<br>")
    return html_out, matched


def risk_level(prob: float):
    """Map a fake-probability to a (label, streamlit_color) pair.

    Streamlit's markdown color syntax supports: blue, green, orange,
    red, violet, gray/grey, rainbow, primary.
    """
    if prob >= 0.75:
        return "High Risk", "red"
    elif prob >= 0.5:
        return "Moderate Risk", "orange"
    elif prob >= 0.25:
        return "Low Risk", "blue"
    else:
        return "Minimal Risk", "green"
