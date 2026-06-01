# 🎬 חיזוי דירוג IMDb לפני יציאת סרט

מטלת סיכום חלק 2 — Machine Learning.
חיזוי `averageRating` (1–10) של סרטים מתוך נתונים שידועים **לפני יציאת הסרט**.

---

## 📊 תוצאות (10-Fold Cross-Validation)

| מודל | RMSE | MAE | R² |
|------|------|-----|-----|
| **Elastic Net** (מוגש לתחרות) | 1.0807 ± 0.0083 | 0.8121 ± 0.008 | 0.4025 ± 0.0079 |
| Random Forest (לניתוחים בסעיפים 5+) | 1.0483 ± 0.0084 | 0.7852 ± 0.0072 | 0.4379 ± 0.0061 |

* **המודל המוגש לתחרות**: Elastic Net.
* **המודל לניתוחים** (Error/Fairness/Feature Importance): Random Forest.

---

## 📁 מבנה הקבצים בהגשה

```
notebook.ipynb              ← המחברת המלאה
model.pkl                   ← Pipeline של Elastic Net (ישיר, לא bundle!)
enrichment_globals.pkl      ← מילוני העשרה (נטענים אוטומטית ע"י prepare_data)
enriched_dataset.csv        ← הדאטה אחרי העשרת TMDB (לקיצור זמן ריצה)
dataset.csv                 ← הדאטה המקורי
README.md                   ← הקובץ הזה
requirements.txt            ← תלויות (קריטי: scikit-learn==1.6.1)
AI_usage_log.md             ← תיעוד שימוש ב-AI
report.pdf                  ← דוח 6-8 עמודים
```

---

## ⚡ הפעלת קוד הבדיקה של המרצה

הקוד יעבוד **בדיוק כמו שכתבת במייל**, ללא שינויים:

```python
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("train.csv")
y = df["averageRating"]
X = prepare_data(df)              # אוטומטית טוען enrichment_globals.pkl

try:
    model = joblib.load("model.pkl")
except Exception:
    import pickle
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

y_pred = model.predict(X)         # ✅ עובד — model.pkl הוא Pipeline ישיר

metrics = pd.DataFrame([{
    "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
    "MAE":  mean_absolute_error(y, y_pred),
    "R2":   r2_score(y, y_pred),
}])
```

**חשוב למרצה**: יש להעתיק את שני הקבצים `model.pkl` ו-`enrichment_globals.pkl`
לאותה תיקייה. `prepare_data` תטען את ה-enrichment אוטומטית.

---

## ⚡ הפעלה — דרך מהירה (~1.5-2 שעות) ⭐ מומלץ

```bash
ls dataset.csv enriched_dataset.csv notebook.ipynb
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

המחברת מזהה אוטומטית את `enriched_dataset.csv` ומדלגת על TMDB API.

---

## 🔒 מניעת Data Leakage

### עמודות "אסורות" (אינן פיצ'רים ישירים):

| עמודה | סטטוס | סיבה |
|--------|--------|------|
| `averageRating` | 🔴 leakage | המשתנה התלוי |
| `numVotes` | 🔴 leakage | זמין רק אחרי יציאה |
| `BoxOffice` | 🔴 leakage | הכנסות אחרי יציאה |

### סינון זמני בפיצ'רים היסטוריים

כל פיצ'ר היסטורי (`feat_actor_hist_avg`, וכו') מסונן `year < target_year`.
**נאמת ב-Cell 38 ב-`assert`**.

### עיבוד מקדים בתוך ה-Pipeline

`Imputer / Scaler / TargetEncoder / TF-IDF` כולם **בתוך** ה-Pipeline,
ומותאמים רק על fold האימון בכל קיפול של ה-CV.

### ניקוי plot אגרסיבי

* footnote markers `[1][2][3]` הוסרו (26,883 מקרים)
* משפטים עם דפוסי פוסט-יציאה הוסרו ("won X Award", "cult classic", "highest grossing")
* Wikipedia-style openers הוסרו
* רק 2.7% מהplots הפכו ל-NaN, אבל leakage כמעט מלא הוסר

---

## 🧩 ארכיטקטורת הקוד

### prepare_data() — עצמאית לחלוטין

```python
def prepare_data(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Self-contained: on first call, auto-loads enrichment_globals.pkl
    if not in memory. Returns engineered features ready for model.predict().
    """
```

* **בתוך המחברת**: משתמשת ב-globals הקיימים.
* **מחוץ למחברת**: טוענת `enrichment_globals.pkl` אוטומטית.

### model.pkl — Pipeline ישיר

```python
joblib.load("model.pkl")  # → Pipeline object (לא dict!)
```

המודל הוא `Pipeline([preprocessor, ElasticNet])` ישיר. הקוד של המרצה
`model.predict(X)` עובד מיד.

---

## 🧩 פיצ'רים (71 בסך הכל)

* היסטוריות שחקנים/במאי/כותב/מפיק (13 פיצ'רים, מסוננים `year<target`)
* תוכן וטקסט: `plot_svd_0..79` (TF-IDF + SVD)
* זמן וז'אנר: 6 פיצ'רים
* תקציב: log_budget + is_missing_budget
* קטגוריאליים: Country, Language, certification, release_season (Target Encoding)
* בינארי, כותרת, אינטראקציות, one-hot ז'אנרים, missing indicators

---

## 📡 מקורות נתונים

* **`dataset.csv`** — הדאטה הניתן ע"י המרצה.
* **IMDb non-commercial datasets** — לבניית ההיסטוריות.
* **TMDB API** — לעמודות `budget_usd`, `certification`, `release_month`.


