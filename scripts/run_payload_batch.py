import json
from pathlib import Path
import requests

API = "http://127.0.0.1:8000/preauth/decision"
DIR = Path("tmp_payloads")

files = sorted([p for p in DIR.glob("sample_*.json")])
if not files:
    raise SystemExit("No payloads found. Generate them first.")

def true_label_from_filename(name: str) -> int:
    # sample_legit_* => 0, sample_fraud_* => 1
    return 1 if "fraud" in name else 0

counts = {
    "legit": {"APPROVE": 0, "REVIEW": 0, "BLOCK": 0},
    "fraud": {"APPROVE": 0, "REVIEW": 0, "BLOCK": 0},
}

false_neg = []  # fraud approved
false_pos = []  # legit blocked

for p in files:
    payload = json.loads(p.read_text())
    r = requests.post(API, json=payload, timeout=10)
    if r.status_code != 200:
        print(f"{p.name}: HTTP {r.status_code} -> {r.text}")
        continue

    out = r.json()
    decision = out.get("decision", "UNKNOWN")
    y = true_label_from_filename(p.name)
    group = "fraud" if y == 1 else "legit"

    if decision not in counts[group]:
        decision = "UNKNOWN"
    else:
        counts[group][decision] += 1

    if y == 1 and decision == "APPROVE":
        false_neg.append((p.name, out.get("scores", {}).get("xgb_probability")))
    if y == 0 and decision == "BLOCK":
        false_pos.append((p.name, out.get("scores", {}).get("xgb_probability")))

print("\n=== SUMMARY ===")
print("LEGIT:", counts["legit"])
print("FRAUD:", counts["fraud"])

print("\n=== False negatives (fraud approved) ===")
for name, prob in false_neg[:10]:
    print(f"- {name}: p_xgb={prob}")

print("\n=== False positives (legit blocked) ===")
for name, prob in false_pos[:10]:
    print(f"- {name}: p_xgb={prob}")
