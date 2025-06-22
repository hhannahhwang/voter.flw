#!/usr/bin/env python3
import glob, joblib, numpy as np, pandas as pd
import shap
from pathlib import Path
from task import build_xy
import warnings
warnings.filterwarnings('ignore')

# 1) load model
bundle = joblib.load("fed_turnout.joblib")
model  = bundle["model"]
feature_names = bundle["features"]

def get_top_features(importance_scores, feature_names, top_n=10):
    """Get top N most important features with their scores."""
    indices = np.argsort(importance_scores)[::-1][:top_n]
    return [(feature_names[i], importance_scores[i]) for i in indices]

rows, raw_risks = [], []
county_shap_data = {}

# 2) compute raw risk
for csv_path in sorted(glob.glob("result/*.csv")):
    county = Path(csv_path).stem.replace("_Merged", "")
    X, _   = build_xy(csv_path)
    probs  = model.predict_proba(X)[:, 1]

    mean_p = probs.mean()
    risk_raw = (1.0 - mean_p) * -1 + 1          # flip risk (* -1 then add 1)
    rows.append([county, len(probs), mean_p, risk_raw])
    raw_risks.append(risk_raw)
    
    # Compute SHAP values
    background_size = min(100, len(X))
    background_indices = np.random.choice(len(X), background_size, replace=False)
    background_data = X[background_indices]
    
    if len(X) <= 1000:
        explainer = shap.KernelExplainer(model.predict_proba, background_data)
        explain_size = min(50, len(X))
        explain_indices = np.random.choice(len(X), explain_size, replace=False)
        explain_data = X[explain_indices]
        shap_values = explainer.shap_values(explain_data)
    else:
        explainer = shap.PermutationExplainer(model.predict_proba, background_data)
        explain_size = min(100, len(X))
        explain_indices = np.random.choice(len(X), explain_size, replace=False)
        explain_data = X[explain_indices]
        shap_values = explainer.shap_values(explain_data)
    
    # For binary classification, get positive class SHAP values
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    
    # Calculate feature importance for this county
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Store county-specific SHAP analysis
    county_shap_data[county] = {
        'feature_importance': mean_abs_shap,
        'top_features': get_top_features(mean_abs_shap, feature_names, top_n=10)
    }

# 3) min-max scale
rmin, rmax = min(raw_risks), max(raw_risks)
def scale(r): return (r - rmin) / (rmax - rmin) if rmax != rmin else 0.0

for row in rows:
    row.append(scale(row[3]))        # append scaled risk

# 4) dashboard
cols = ["County", "Voters", "Avg_P_vote", "Risk_raw", "Risk_scaled"]
dash = (pd.DataFrame(rows, columns=cols)
          .sort_values("Risk_scaled", ascending=False)
          .reset_index(drop=True))

print("\nCounty-Risk Dashboard (scaled 0-1)")
print(dash[["County", "Risk_scaled"]].to_string(index=False, float_format=lambda x:f"{x:.3f}"))
dash.to_csv("county_risk_dashboard_scaled.csv", index=False)
print("\nWrote county_risk_dashboard_scaled.csv")

# Print SHAP analysis for each county
print("\nSHAP Analysis by County")

for _, row in dash.iterrows():
    county = row['County']
    risk_scaled = row['Risk_scaled']
    
    if county in county_shap_data:
        print(f"\n{county.upper()} - Risk Score: {risk_scaled:.3f}")
        print("Top 10 Most Important Features:")
        
        top_features = county_shap_data[county]['top_features']
        for i, (feature, importance) in enumerate(top_features, 1):
            print(f"{i:2d}. {feature:<40} {importance:.4f}")