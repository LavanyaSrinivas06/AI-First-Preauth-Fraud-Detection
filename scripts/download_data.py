# scripts/download_data.py
from pathlib import Path

RAW_DIR = Path("data/raw")
DATA_FILE = RAW_DIR / "creditcard.csv"

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_FILE.exists():
        size_mb = DATA_FILE.stat().st_size / (1024 * 1024)
        print(f"Found {DATA_FILE} (~{size_mb:.1f} MB)")
        if size_mb < 140:
            print("⚠️ File seems too small (<140 MB). Re-download to avoid corrupt CSV.")
        else:
            print("✅ Dataset looks OK.")
        return

    print("❌ creditcard.csv not found.")
    print("Please download manually from Kaggle:")
    print("  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print("Steps:")
    print("  1) Download the dataset ZIP")
    print("  2) Extract `creditcard.csv`")
    print("  3) Place it at: data/raw/creditcard.csv")
    print("Optional: verify file size is > 140 MB.")

if __name__ == "__main__":
    main()
