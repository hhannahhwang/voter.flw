import os
import glob
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai

print("🔁 Switching to Gemini API")

# Load & configure
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ GEMINI_API_KEY missing"); exit(1)
genai.configure(api_key=api_key)

print("✅ Gemini API key loaded")

# Load articles
with open("articles/articles.txt", "r", encoding="utf-8") as f:
    raw = f.read()
    articles = [a.strip() for a in raw.split("===") if a.strip()]
print(f"📰 {len(articles)} articles loaded")

# Process each county
for csv_path in glob.glob("data/*_fincsv.csv"):
    county = os.path.basename(csv_path).replace("_fincsv.csv", "")
    out_path = f"data/{county}_analysis.txt"
    if os.path.exists(out_path):
        print(f"⏩ Skipping {county} — analysis already exists.")
        continue

    print(f"▶️ Processing {county}")
    df = pd.read_csv(csv_path, header=None, names=["feature", "importance"])
    risk_score_row = df.iloc[0]
    if risk_score_row["feature"] == "RISK_SCORE":
        risk_score = float(risk_score_row["importance"])
        df = df.iloc[1:]  # Drop the risk score row
    else:
        print(f"❌ {county}: Missing risk score row")
        continue

    features = dict(zip(df.feature, df.importance))
    prompt = (
        f"You are a political data scientist analyzing potential voter suppression in {county} County.\n\n"
        f"The overall voter suppression risk score for this county is {risk_score:.3f}.\n"
        "Interpret it using this scale:\n"
        "- Below 0.35 = ⚠️ High Risk\n"
        "- 0.35–0.75 = ⚠️ Medium Risk\n"
        "- Above 0.75 = ✅ Low Risk\n\n"
        "You are also given the top model features and factual findings.\n"
        "Analyze why these features may contribute to suppression risk, using evidence from the findings.\n"
        "Avoid generic phrasing and do not mention 'articles'.\n\n"
        "Return your output as:\n"
        "1. A line stating the overall suppression risk level (e.g. '⚠️ High Risk')\n"
        "2. 3–5 bullet points, each starting with a severity tag:\n"
        "   - ⚠️ High — strong evidence of targeted or systemic suppression\n"
        "   - ⚠️ Medium — some evidence, warrants monitoring\n"
        "   - ✅ Low — minimal evidence or unlikely impact\n\n"
        "Each bullet must be under 15 words and explain the significance of a specific feature.\n\n"
        f"Top Features:\n{chr(10).join([f'{k}: {v:.6f}' for k, v in features.items()])}\n\n"
        f"Factual Findings:\n{chr(10).join([f'{a}' for a in articles])}\n\n"
        f"Output:"
    )


    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        resp = model.generate_content(prompt)
        output = resp.text
        out_path = f"data/{county}_analysis.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"✅ Saved to {out_path}")
    except Exception as e:
        print(f"❌ Error with Gemini for {county}: {e}")
