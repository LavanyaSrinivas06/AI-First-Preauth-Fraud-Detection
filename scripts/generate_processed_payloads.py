import json
from pathlib import Path
import pandas as pd

DATA = Path("data/processed/val.csv")
OUT = Path("tmp_payloads")
OUT.mkdir(exist_ok=True)

N_LEGIT = 15
N_FRAUD = 15
SEED = 42

df = pd.read_csv(DATA)
if "Class" not in df.columns:
    raise SystemExit("val.csv must contain 'Class' column")

# exact processed feature list (102)
feature_cols = [c for c in df.columns if c != "Class"]
if len(feature_cols) != 102:
    raise SystemExit(f"Expected 102 features, got {len(feature_cols)}")

legit = df[df["Class"] == 0].sample(N_LEGIT, random_state=SEED)
fraud = df[df["Class"] == 1].sample(N_FRAUD, random_state=SEED + 1)

def save_rows(rows: pd.DataFrame, prefix: str):
    for i, (_, r) in enumerate(rows.iterrows(), 1):
        payload = {c: (None if pd.isna(r[c]) else r[c]) for c in feature_cols}
        # keep numeric JSON clean
        path = OUT / f"{prefix}_{i}.json"
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"saved: {path}")

save_rows(legit, "sample_legit")
save_rows(fraud, "sample_fraud")

print("done.")
