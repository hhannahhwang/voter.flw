import pandas as pd, pathlib as p

data_dir = p.Path("data")
votehistory_dir = data_dir / "votehistory"
result_dir = data_dir / "result"
result_dir.mkdir(exist_ok=True)

county_files = list(votehistory_dir.glob("*.txt"))
counties = [f.stem for f in county_files]

for cslug in counties:
    try:
        abs16 = pd.read_csv(f"data/absentee2016/absentee_{cslug}_20161108.csv", dtype=str, encoding="latin1")
        abs20 = pd.read_csv(f"data/absentee2020/{cslug}_absentee_20201103.csv", dtype=str, encoding="latin1")
        abs24 = pd.read_csv(f"data/absentee2024/{cslug}_absentee_20241105.csv", dtype=str, encoding="latin1")
        hist  = pd.read_csv(f"data/votehistory/{cslug}.txt", sep="\t", dtype=str, encoding="latin1")

        for df in (abs16, abs20, abs24, hist):
            df["voter_reg_num"] = df["voter_reg_num"].str.zfill(12)

        voters = pd.concat([abs24, abs20, abs16], ignore_index=True).drop_duplicates("voter_reg_num")

        keep = ["voter_reg_num","county_desc","race","gender","age",
                "voter_city","voter_state","voter_zip","voter_party_code"]
        voters = voters[keep]

        def flag(df, date):
            ids = df[df["election_lbl"]==date]["voter_reg_num"]
            return voters["voter_reg_num"].isin(ids).astype(int)

        voters["voted_2016"] = flag(hist,"11/08/2016")
        voters["voted_2020"] = flag(hist,"11/03/2020")
        voters["voted_2024"] = flag(hist,"11/05/2024")

        outfile = f"data/result/{cslug}_Merged.csv"
        voters.to_csv(outfile, index=False)
        print(voters.head())
    
    except FileNotFoundError as e:
        print(f"File not found for county {cslug}: {e}")
    except Exception as e:
        print(f"Error processing {cslug}: {e}")