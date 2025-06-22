import os
import glob

input_dir = "raw"
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

for path in glob.glob(os.path.join(input_dir, "*_raw.txt")):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    county = None
    risk_score = None
    features = []

    for line in lines:
        line = line.strip()

        if " - Risk Score:" in line:
            # Match: "ALEXANDER - Risk Score: 0.991"
            parts = line.split(" - Risk Score:")
            county = parts[0].strip().upper()
            risk_score = float(parts[1].strip())

        elif line and any(line.startswith(f"{i}.") for i in range(1, 11)):
            # Match feature lines: "1. voted_2020_w 0.0280"
            parts = line.split()
            if len(parts) >= 3:
                feature = parts[1]
                value = float(parts[-1])
                features.append((feature, value))

    if county and risk_score is not None:
        out_path = os.path.join(output_dir, f"{county}_fincsv.csv")
        with open(out_path, "w", encoding="utf-8") as out:
            out.write(f"RISK_SCORE,{risk_score:.3f}\n")
            for feat, imp in features:
                out.write(f"{feat},{imp:.6f}\n")
        print(f"✅ Converted: {out_path}")
    else:
        print(f"❌ Failed to process {path}")
