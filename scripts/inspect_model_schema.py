from pathlib import Path
import joblib
import pandas as pd
import json

# -----------------------------
# Paths (adjust only if needed)
# -----------------------------
TRAIN_PATH = Path("data/processed/train.csv")
VAL_PATH = Path("data/processed/val.csv")
TEST_PATH = Path("data/processed/test.csv")

FEATURES_JSON = Path("artifacts/features.json")
PREPROCESS_PATH = Path("artifacts/preprocess.joblib")

TARGET_COL = "Class"

print("\n==============================")
print(" MODEL SCHEMA INSPECTION ")
print("==============================\n")

# -----------------------------
# 1. From processed CSVs
# -----------------------------
print("1️⃣ Checking processed datasets\n")

df = pd.read_csv(TRAIN_PATH)

print(f"Train shape: {df.shape}")

if TARGET_COL not in df.columns:
    raise RuntimeError(f"Target column '{TARGET_COL}' not found in train.csv")

feature_cols = [c for c in df.columns if c != TARGET_COL]

print(f"\nTarget column: {TARGET_COL}")
print(f"Number of features: {len(feature_cols)}")
print("\nFeature columns (ORDER MATTERS):")
for i, c in enumerate(feature_cols):
    print(f"{i+1:02d}. {c}")

# -----------------------------
# 2. Check consistency across splits
# -----------------------------
print("\n2️⃣ Checking schema consistency across train/val/test\n")

def get_features(path: Path):
    d = pd.read_csv(path)
    return [c for c in d.columns if c != TARGET_COL]

train_feats = get_features(TRAIN_PATH)
val_feats = get_features(VAL_PATH)
test_feats = get_features(TEST_PATH)

assert train_feats == val_feats == test_feats, "❌ Feature mismatch across splits"

print("✅ Train / Val / Test schemas are IDENTICAL")

# -----------------------------
# 3. From features.json (if exists)
# -----------------------------
print("\n3️⃣ Checking artifacts/features.json\n")

if FEATURES_JSON.exists():
    with open(FEATURES_JSON, "r") as f:
        meta = json.load(f)

    feats = meta.get("feature_names")
    if feats:
        print(f"features.json lists {len(feats)} features")
        print("First 10 features:", feats[:10])
        print("Last 10 features:", feats[-10:])

        if feats == feature_cols:
            print("✅ features.json EXACTLY matches training data schema")
        else:
            print("⚠️ features.json does NOT match training CSV order")
    else:
        print("⚠️ features.json found but no 'feature_names' key")
else:
    print("ℹ️ features.json not found (this is OK)")

# -----------------------------
# 4. From preprocessing pipeline
# -----------------------------
print("\n4️⃣ Checking preprocessing pipeline\n")

if PREPROCESS_PATH.exists():
    preprocess = joblib.load(PREPROCESS_PATH)
    try:
        names = preprocess.get_feature_names_out()
        print(f"Pipeline exposes {len(names)} feature names")
        print("First 10:", names[:10])
        print("Last 10:", names[-10:])
    except Exception as e:
        print("⚠️ Could not extract feature names from pipeline:", e)
else:
    print("ℹ️ preprocess.joblib not found")

print("\n==============================")
print(" SCHEMA INSPECTION COMPLETE ")
print("==============================\n")
