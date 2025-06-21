import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("alexanderMerged.csv", dtype={"voter_reg_num": str})

base_cols = [
    "race", "gender", "county_desc", "voter_party_code",
    "voter_zip", "age", "voted_2016", "voted_2020"
]
X = pd.get_dummies(df[base_cols], dummy_na=True)

# weight party dummies
for col in [c for c in X.columns if "voter_party_code_" in c]:
    X[f"{col}_w"] = X[col] * 2

# weight voted_2016
if "voted_2016" in X.columns:
    X["voted_2016_w"] = X["voted_2016"] * 1.5

# weight voted_2020
if "voted_2020" in X.columns:
    X["voted_2020_w"] = X["voted_2020"] * 0.75

y = df["voted_2024"]

# ─── Train / test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.25, random_state=42
)

# ─── Random‑Forest training ───────────────────────────────────────────────────
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
clf.fit(X_train, y_train)

# ─── Accuracy ─────────────────────────────────────────────────────────────────
print(f"Accuracy: {report['accuracy']:.3f}")
print(f"Precision (voted): {report['1']['precision']:.3f}")
print(f"Recall (voted): {report['1']['recall']:.3f}")
print(f"F1-score (voted): {report['1']['f1-score']:.3f}")

# ─── Top‑15 feature importances ───────────────────────────────────────────────
feature_importances = pd.Series(clf.feature_importances_, index=X.columns)
top15 = feature_importances.sort_values(ascending=False).head(15)
print("\nTop 15 Important Features:")
print(top15)
