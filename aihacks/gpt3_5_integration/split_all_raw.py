import os
import re

input_path = "all_raw.txt"
output_dir = "raw"
os.makedirs(output_dir, exist_ok=True)

with open(input_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

current_lines = []
county_name = None

for line in lines:
    # Detect county start
    match = re.match(r"^([A-Z\s\-]+)\s*-\s*Risk Score:\s*(\d+\.\d+)", line)
    if match:
        # If we were processing a previous county, save it
        if county_name and current_lines:
            out_path = os.path.join(output_dir, f"{county_name.lower().strip().replace(' ', '_')}_raw.txt")
            with open(out_path, "w", encoding="utf-8") as out_file:
                out_file.writelines(current_lines)
            print(f"✅ Saved {county_name} to {out_path}")
            current_lines = []

        # Start new county
        county_name = match.group(1).strip()
    
    if county_name:
        current_lines.append(line)

# Save the last county
if county_name and current_lines:
    out_path = os.path.join(output_dir, f"{county_name.lower().strip().replace(' ', '_')}_raw.txt")
    with open(out_path, "w", encoding="utf-8") as out_file:
        out_file.writelines(current_lines)
    print(f"✅ Saved {county_name} to {out_path}")