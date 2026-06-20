# 🛡️ Fake Job Posting Detector

This is a machine learning system that detects fraudulent job postings using a combination of text analysis and structured feature engineering. It compares two models side-by-side — a classical TF-IDF + Logistic Regression baseline and a fine-tuned DistilBERT transformer — and explains *why* a posting looks suspicious, not just whether it does.

Built as an end-to-end ML project: EDA → feature engineering → baseline modeling → transformer fine-tuning → deployment, using the [Kaggle Fake Job Postings dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction).

---

## 🎯 Live Demo

> 

---

## 📊 Model Performance

| Model | F1-Score (Fake class) | Precision (Fake) | Recall (Fake) |
|---|---|---|---|
| **Baseline** — TF-IDF + Structured Features + Logistic Regression | 0.82 | 0.74 | 0.91 |
| **DistilBERT** — Fine-tuned transformer (structured-aware text format) | **0.91** | 0.96 | 0.86 |

The DistilBERT model is hosted on [HuggingFace Hub](https://huggingface.co/Yuvraj-Bansal/fake_job-distilbert) and loaded at runtime — no large weight files are committed to this repo. Its decision threshold (0.55) was tuned via a sweep over the validation set to optimize F1, rather than using the default 0.5 cutoff.

---

## 🔍 Key EDA Findings

| # | Finding | Why It Matters |
|---|---|---|
| 1 | **~4.8% of postings are fake** (17,880 total) | Severe class imbalance → accuracy is misleading; F1-score used instead |
| 2 | **No company profile → 17.7% fake rate** vs 1.9% when present | Omission of structured fields is a stronger signal than text content alone |
| 3 | **No company logo → 15.9% fake rate** vs 2.0% when present | Scammers skip legitimacy markers; *what's absent* predicts fraud |
| 4 | **Description length barely differs** (real: 1,221 chars, fake: 1,155) | Text *content* matters far more than length for classification |
| 5 | **Fake postings cluster in Oil & Energy, Admin, Engineering roles** | Scammers target high-prestige or high-desperation job seekers |

These findings directly shaped the model design:
- **SMOTE** used in the baseline to handle class imbalance (DistilBERT uses class-weighted loss instead)
- **Structured features** (has_logo, has_company_profile, etc.) included alongside TF-IDF text features in the baseline
- **F1-score** (fake class) used as the primary evaluation metric throughout, not accuracy


---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      Streamlit App                       │
│              (job title, description, flags)             │
└────────────────────┬────────────────────┬────────────────┘
                     │                    │
            ┌────────▼─────────┐  ┌───────▼─────────┐
            │   Baseline Path  │  │ DistilBERT Path │
            │                  │  │                 │
            │  clean_text()    │  │ build_distilbert│
            │  TF-IDF vectorize│  │ _text()         │
            │  + structured    │  │ tokenize        │
            │  features        │  │ forward pass    │
            │  Logistic Reg.   │  │ (HF Hub model)  │
            └────────┬─────────┘  └────────┬────────┘
                     │                     │
                     ▼                     ▼
              Fake probability    Fake probability
                     │                     │
                     └─────────┬───────────┘
                               ▼
                    Risk level + red flags
                    + highlighted text
                    + safety tips
```

### Two models, two different strengths
- **Baseline (TF-IDF + LR)**: fast, interpretable, runs anywhere with no GPU. Sees structured features (logo, profile, salary presence) explicitly as numeric inputs.
- **DistilBERT**: captures semantic and contextual patterns in language that keyword-based methods miss. Trained on a structured-aware text format (`Title: ... [SEP] Profile: ... [SEP] Requirements: ... [SEP] Description: ...`) so it implicitly learns from field presence/absence too.

---

## 📁 Project Structure

```
Fake_Job/
├── app.py                          # Streamlit app — both models side by side
├── requirements.txt
├── notebooks/
│   ├── EDA.ipynb                   # Exploratory data analysis
│   ├── Baseline_Model.ipynb        # TF-IDF + Logistic Regression
│   └── DistilBERT_Model.ipynb      # DistilBERT fine-tuning (Colab T4)
├── src/
│   ├── preprocess.py               # Text cleaning (matches training exactly)
│   ├── features.py                 # Structured feature extraction + keyword lists
│   ├── model.py                    # Model loading & inference (both models)
│   ├── explainer.py                # Red flag detection, highlighting, risk levels
│   ├── lr_model.pkl                # Baseline: trained Logistic Regression
│   ├── tfidf.pkl                   # Baseline: fitted TF-IDF vectorizer
│   └── scaler.pkl                  # Baseline: fitted feature scaler
└── data/
    └── fake_job_postings.csv
```

DistilBERT's weights are **not** stored in this repo — they live on [HuggingFace Hub](https://huggingface.co/Yuvraj-Bansal/fake_job-distilbert) and are downloaded at runtime, keeping the repo lightweight.

---

## ⚙️ Setup & Local Run

```bash
git clone https://github.com/Yuvraj-Bansal/Fake-Job-Postings.git
cd Fake_Job_Postings

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt

streamlit run app.py
```

First run will download the DistilBERT model (~250MB) from HuggingFace Hub — subsequent runs use the cached version.

---

## 🧠 Tech Stack

- **Language/Core:** Python, Pandas, NumPy
- **Baseline Model:** scikit-learn (TF-IDF, Logistic Regression, SMOTE)
- **Transformer:** HuggingFace `transformers`, PyTorch, DistilBERT (fine-tuned)
- **Training:** Google Colab (T4 GPU), mixed-precision (AMP), dynamic padding
- **Deployment:** Streamlit, HuggingFace Hub (model hosting)
- **Versioning:** Git, Git LFS (baseline artifacts)

---

## 📌 Notes on Methodology

- **Train/serve consistency**: the exact same `clean_text()` and `extract_structured_features()` functions used during training are reused at inference time in `src/`, to avoid any mismatch between how the model was trained and how it's queried.
- **Class imbalance**: handled via SMOTE for the baseline and class-weighted cross-entropy loss (with softened weights via square-root scaling) for DistilBERT.
- **Threshold tuning**: DistilBERT's classification threshold was tuned on the validation set (optimal: 0.55) rather than assumed at 0.5.
- **DistilBERT input format**: structured fields are folded into the text itself (`Title: ... [SEP] Profile: ... [SEP] ...`) so the transformer can implicitly learn from field presence/absence, similar to the explicit structured features used in the baseline.

*Built by Yuvraj Bansal*
