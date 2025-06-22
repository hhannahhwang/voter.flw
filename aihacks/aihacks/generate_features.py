# generate_features.py
import pandas as pd
import json

df = pd.read_csv("result/alamance_Merged.csv", dtype={"voter_reg_num": str})
base_cols = [
    "race", "gender", "county_desc", "voter_party_code",
    "voter_zip", "age", "voted_2016", "voted_2020"
]
X = pd.get_dummies(df[base_cols], dummy_na=True)
for col in [c for c in X.columns if "voter_party_code_" in c]:
    X[f"{col}_w"] = X[col] * 2
X["voted_2016_w"] = df["voted_2016"] * 1.5
X["voted_2020_w"] = df["voted_2020"] * 0.75

with open("feature_names.json", "w") as f:
    json.dump(list(X.columns), f)
