## Data Sources & Schema

- **Base dataset:** Kaggle Credit Card Fraud — https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  
  Place the extracted file at `data/raw/creditcard.csv`.  
  Helper: `python scripts/download_data.py` (checks presence/size).

- **Data dictionary:** `docs/data_dictionary.md` (base + enrichment fields)

- **Enriched schema (machine-readable):** `docs/schema_enriched.json`

> Note: Enrichment fields will be generated programmatically in later tickets (device, network, behavior, and geo features).

