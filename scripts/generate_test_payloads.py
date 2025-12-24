import json
import pandas as pd
from pathlib import Path

DATA = Path("data/processed")
OUT = Path("tmp_payloads")
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA / "val.csv")

# schema from file (source of truth)
FEATURES = [c for c in df.columns if c != "Class"]
assert len(FEATURES) == 102, f"Expected 102 features, got {len(FEATURES)}"

def save_rows(rows: pd.DataFrame, prefix: str):
    for i, (_, r) in enumerate(rows.iterrows(), 1):
        payload = {k: (None if pd.isna(r[k]) else r[k]) for k in FEATURES}
        path = OUT / f"{prefix}_{i}.json"
        with open(path, "w") as f:
            json.dump(payload, f)
        print(f"saved: {path}")

# pick rows (val is fine)
approve_rows = df[df["Class"] == 0].sample(10, random_state=42)
fraud_rows   = df[df["Class"] == 1].sample(20, random_state=99)

save_rows(approve_rows, "sample_legit")
save_rows(fraud_rows, "sample_fraud")
