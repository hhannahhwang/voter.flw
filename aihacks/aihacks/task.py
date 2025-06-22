import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
import json, os

FEATURES_PATH = "feature_names.json"

def build_xy(csv_path: str):
    df = pd.read_csv(csv_path, dtype={"voter_reg_num": str})

    base_cols = [
        "race", "gender", "county_desc", "voter_party_code",
        "voter_zip", "age", "voted_2016", "voted_2020"
    ]
    X = pd.get_dummies(df[base_cols], dummy_na=True)

    for c in [c for c in X.columns if "voter_party_code_" in c]:
        X[f"{c}_w"] = X[c] * 2
    X["voted_2016_w"] = X["voted_2016"] * 1.5
    X["voted_2020_w"] = X["voted_2020"] * 1.5

    if os.path.exists(FEATURES_PATH):
        feat_order = json.load(open(FEATURES_PATH))
        for col in feat_order:
            if col not in X.columns:
                X[col] = 0
        X = X[feat_order]
    else:
        json.dump(list(X.columns), open(FEATURES_PATH, "w"))

    y = df["voted_2024"].values
    X = X.fillna(0)
    return X.values.astype(np.float32), y.astype(np.int8)

# task.py
from sklearn.neural_network import MLPClassifier


def make_model(n_features: int):
    return MLPClassifier(
        hidden_layer_sizes=(64,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=1,
        warm_start=True
    )


