# 🎬 IMDb Rating Prediction — Pre-Release Movie Forecasting

**Machine Learning Final Project — Part 2**
Predicting a film's `averageRating` (1–10) using only features available **before theatrical release**.

---

## 📊 Results (10-Fold Cross-Validation)

| Model | RMSE | MAE | R² |
|:------|:----:|:---:|:--:|
| **Elastic Net** *(submitted to competition)* | **1.0705** ± 0.0099 | 0.8090 ± 0.0071 | 0.4134 ± 0.0123 |
| Random Forest *(used for §5–§7 analyses)* | **1.0502** ± 0.0090 | 0.7930 ± 0.0054 | 0.4355 ± 0.0082 |

- **Competition model:** Elastic Net — chosen for stability and interpretability (with logit target transformation).
- **Analysis model:** Random Forest — used in Error Analysis, Fairness, and Feature Importance per assignment spec (better-performing model on RMSE).

---

## 📁 Submission Contents

```
notebook.ipynb              ← Full Jupyter notebook (end-to-end pipeline)
model.pkl                   ← Elastic Net Pipeline (direct, not a bundle)
enrichment_globals.pkl      ← Enrichment dictionaries (auto-loaded by prepare_data)
enriched_dataset.csv        ← Pre-enriched data (skip TMDB API, saves ~5h)
dataset.csv                 ← Original dataset from Part 1
report.pdf                  ← 6–8 page report
requirements.txt            ← Dependencies
AI_usage_log.md             ← AI consultation log
README.md                   ← This file
```

---

## ⚡ Running the Instructor's Test Code

The submission is designed to work **as-is** with the exact code snippet from the assignment:

```python
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("train.csv")
y  = df["averageRating"]
X  = prepare_data(df)                  # auto-loads enrichment_globals.pkl

try:
    model = joblib.load("model.pkl")
except Exception:
    import pickle
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

y_pred = model.predict(X)              # ✅ model.pkl is a Pipeline — works directly

metrics = pd.DataFrame([{
    "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
    "MAE":  mean_absolute_error(y, y_pred),
    "R2":   r2_score(y, y_pred),
}])
```

> **Important:** Both `model.pkl` and `enrichment_globals.pkl` must be in the same directory.
> `prepare_data` automatically loads the enrichment file on first call when running outside the notebook.

---

## 🚀 Quick Run (~1.5–2 hours, **recommended**)

```bash
# 1. Verify files
ls dataset.csv enriched_dataset.csv notebook.ipynb

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch notebook
jupyter notebook notebook.ipynb
```

The notebook automatically detects `enriched_dataset.csv` and **skips the TMDB API enrichment** (which would otherwise take 4–6 hours).

---

## 🔒 Data Leakage Prevention

### Forbidden Columns (excluded from features)

| Column | Status | Reason |
|:-------|:------:|:-------|
| `averageRating` | 🔴 leakage | The target variable |
| `numVotes` | 🔴 leakage | Accumulates only after release |
| `BoxOffice` | 🔴 leakage | Revenue is post-release |

### Temporal Filtering of Historical Features

Every historical feature (`feat_actor_hist_avg`, `feat_director_hist_avg`, `feat_franchise_avg`, etc.) is filtered to `year < target_year` — meaning a movie's actor-history feature only uses ratings from films released **before** that movie.

### In-Pipeline Preprocessing

`SimpleImputer`, `StandardScaler`, `TargetEncoder`, `TF-IDF + SVD` — all live **inside** the scikit-learn `Pipeline`, so each is fit on the training fold only during cross-validation. No information bleeds from validation folds.

### Aggressive Plot Cleaning

The `clean_plot()` function removes post-release content from plot text:

- Footnote markers `[1][2][3]` (Wikipedia-style references)
- Sentences containing post-release patterns: *"won an Award"*, *"cult classic"*, *"highest-grossing"*, *"received critical acclaim"*
- Wikipedia-style openers like *"X is a YEAR film by..."*

### Plot Ablation Test (Empirical Verification)

To verify no plot-text leakage, we ran a full 10-fold CV without the plot embeddings:

| Configuration | RMSE | Δ |
|:--------------|:----:|:--:|
| EN with plot text | 1.0705 | — |
| EN without plot text | 1.0741 | **+0.0036 (+0.33%)** |

The marginal contribution of plot text (+0.33%) confirms there is no significant leakage — a leaky plot would have given much larger uplift.

---

## 🧩 Code Architecture

### `prepare_data()` — Fully Self-Contained

```python
def prepare_data(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Self-contained: on first call, auto-loads enrichment_globals.pkl
    if the enrichment dicts are not in memory (i.e., running outside
    the notebook). Returns engineered features ready for model.predict().
    """
```

- **Inside the notebook:** uses the global enrichment dicts built during training.
- **Outside the notebook (instructor's environment):** auto-loads `enrichment_globals.pkl`.

### `model.pkl` — Direct Pipeline

```python
joblib.load("model.pkl")   # → sklearn.pipeline.Pipeline (NOT a dict / bundle)
```

The artifact is a `Pipeline([preprocessor, TransformedTargetRegressor(ElasticNet)])` saved directly. The instructor's `model.predict(X)` call works immediately — no unwrapping needed.

---

## 🧪 Feature Set (85 Input Columns)

| Group | Count | Examples |
|:------|:-----:|:---------|
| **Cast histories** (temporally filtered) | 8 | `feat_actor_hist_avg`, `feat_actor_recent_avg`, `feat_n_proven_stars` |
| **Crew histories** | 3 | `feat_director_hist_avg`, `feat_writer_hist_avg`, `feat_best_crew` |
| **Director-specific** | 3 | `feat_director_genre_avg`, `feat_director_rating_std`, `is_missing_director` |
| **Franchise / Collaboration** | 6 | `feat_franchise_avg`, `feat_collab_dir_actor_avg`, `feat_collab_dir_writer_avg` |
| **Genre dynamics** | 3 | `feat_genre_year_trend`, `feat_genre_year_rolling`, `feat_runtime_genre_dev` |
| **Plot text (TF-IDF + SVD)** | 1 input → 80 dims (EN) / 50 dims (RF) | `plot_svd_0..N` |
| **Budget** | 2 | `log_budget`, `is_missing_budget` |
| **Time / season** | 4 | `startYear`, `years_since_2000`, `release_season`, `has_release_date` |
| **Categorical (Target Encoded)** | 4 | `lang_primary`, `country_primary`, `content_rating`, `release_season` |
| **Title-based** | 4 | `title_length`, `title_word_count`, `title_has_colon`, `title_has_number` |
| **Genre one-hot** | 20 | `g_Drama`, `g_Comedy`, `g_Documentary`, … |
| **Polynomial / Interactions** | 14 | `feat_actor_sq`, `actor_x_genre_rolling`, `horror_x_runtime`, … |
| **Missingness flags** | 3 | `is_missing_actor_hist`, `is_missing_director`, `is_missing_writer` |

---

## 📡 Data Sources

| Source | Purpose |
|:-------|:--------|
| `dataset.csv` | Original dataset from Part 1 (provided) |
| IMDb non-commercial datasets | Building actor/director/writer/producer historical features |
| TMDB API | `budget_usd`, `certification` (MPAA), `release_month`, plot, Language, Country |

The TMDB enrichment covers ~70K movies (all titles in the dataset that needed external data).

---

## 🛠️ Technical Requirements

- **Python 3.10+**
- `scikit-learn>=1.5,<2.0` *(model.pkl loads cleanly across this range)*
- `random_state=42` everywhere for reproducibility
- End-to-end runtime: ~1.5–2h with caches (provided) / ~6–9h from scratch

---

*This project was developed in accordance with the academic integrity guidelines specified in the assignment. AI consultation is documented in `AI_usage_log.md`.*
