import json
import glob
import requests
from collections import Counter

URL = "http://127.0.0.1:8000/preauth/decision"

counts = Counter()
files = sorted(glob.glob("tmp_payloads/sample_*.json"))

for f in files:
    with open(f, "r") as fp:
        payload = json.load(fp)
    r = requests.post(URL, json=payload, timeout=10)
    if r.status_code != 200:
        print(f"ERROR {f}: {r.status_code} {r.text[:200]}")
        continue
    decision = r.json().get("decision", "UNKNOWN")
    counts[decision] += 1

print("Decision counts:", dict(counts))
