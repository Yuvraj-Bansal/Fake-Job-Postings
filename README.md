## Key EDA Findings

| # | Finding | Why It Matters |
|---|---|---|
| 1 | **~4.8% of postings are fake** (17,880 total) | Severe class imbalance → accuracy is misleading; F1-score used instead |
| 2 | **No company profile → 17.7% fake rate** vs 1.9% when present | Omission of structured fields is a stronger signal than text content alone |
| 3 | **No company logo → 15.9% fake rate** vs 2.0% when present | Scammers skip legitimacy markers; *what's absent* predicts fraud |
| 4 | **Description length barely differs** (real: 1,221 chars, fake: 1,155) | Text *content* matters far more than length for classification |
| 5 | **Fake postings cluster in Oil & Energy, Admin, Engineering roles** | Scammers target high-prestige or high-desperation job seekers |

These findings directly shaped the model design:
- SMOTE used to handle class imbalance
- Structured features (has_logo, has_company_profile) included alongside TF-IDF text features
- F1-score (fake class) used as the primary evaluation metric, not accuracy
