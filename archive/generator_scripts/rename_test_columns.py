"""
Script to rename columns in test.csv to match expected names for preprocessing pipeline.

- Reads data/processed/test.csv
- Renames columns by removing 'num__', 'cat__' prefixes
- Saves as data/processed/test_renamed.csv
- Safe, does not overwrite original
"""
import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/processed/test.csv")
OUTPUT_PATH = Path("data/processed/test_renamed.csv")

df = pd.read_csv(INPUT_PATH)

# Remove 'num__' and 'cat__' prefixes from column names
new_columns = [col.replace('num__', '').replace('cat__', '') for col in df.columns]
df.columns = new_columns

df.to_csv(OUTPUT_PATH, index=False)
print(f"Renamed columns and saved to {OUTPUT_PATH}")
