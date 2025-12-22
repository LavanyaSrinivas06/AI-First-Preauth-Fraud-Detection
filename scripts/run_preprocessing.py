
from __future__ import annotations

import sys
from pathlib import Path

# Add project root directory to PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import json
from pathlib import Path
import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from src.preprocess.preprocess_pipeline import build_preprocess_pipeline


# ==== CONFIGURATION (easy to change later) ====================================

DATA_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("artifacts")
DOCS_DIR = Path("docs")

ENRICHED_PATH = DATA_DIR / "enriched.csv"

TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "val.csv"
TEST_PATH = DATA_DIR / "test.csv"

TRAIN_NOSMOTE_PATH = DATA_DIR / "train_nosmote.csv"
VAL_NOSMOTE_PATH   = DATA_DIR / "val_nosmote.csv"
TEST_NOSMOTE_PATH  = DATA_DIR / "test_nosmote.csv"


TRAIN_RAW_PATH = DATA_DIR / "train_raw.csv"
VAL_RAW_PATH = DATA_DIR / "val_raw.csv"
TEST_RAW_PATH = DATA_DIR / "test_raw.csv"


PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocess.joblib"
FEATURES_META_PATH = ARTIFACTS_DIR / "features.json"

REPORT_PATH = DOCS_DIR / "preprocessing_summary.md"

TARGET_COL = "Class"   # label column
TIME_COL = "Time"      # column used for time-based splitting

USE_SMOTE = True    # whether to apply SMOTE on training set



# ==== HELPER FUNCTIONS ========================================================

def load_enriched_dataset(sample_n: int | None = None) -> pd.DataFrame:
    """Load the enriched dataset from disk."""
    if not ENRICHED_PATH.exists():
        raise FileNotFoundError(f"Enriched dataset not found at {ENRICHED_PATH}")
    df = pd.read_csv(ENRICHED_PATH)
       # If sampling is enabled, take a random subset
    if sample_n is not None and len(df) > sample_n:
        df = df.sample(sample_n, random_state=42).reset_index(drop=True)
    return df


def make_time_based_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create 70/15/15 time-ordered splits (train, val, test).

    We sort by TIME_COL to avoid temporal leakage:
    - Older transactions -> training
    - Middle period      -> validation
    - Most recent        -> test
    """
    if TIME_COL not in df.columns:
        raise KeyError(f"Time column '{TIME_COL}' not found in dataframe")

    df_sorted = df.sort_values(TIME_COL).reset_index(drop=True)
    n = len(df_sorted)

    train_end = int(0.70 * n)
    val_end = int(0.85 * n)

    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()

    return train_df, val_df, test_df


def identify_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Determine which columns are numerical vs categorical.

    Exclude:
      - TARGET_COL (label)
      - TIME_COL   (used only for ordering, not as a feature)
    """
    exclude_cols = {TARGET_COL, TIME_COL, "device_id"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Categorical: object / category / bool
    cat_cols = df[feature_cols].select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Numerical: everything else that is not categorical and not excluded
    num_cols = [c for c in feature_cols if c not in cat_cols]

    return cat_cols, num_cols


def dataframe_from_transformed(
    X_transformed,
    y: pd.Series,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Turn the transformed matrix + label into a pandas DataFrame.

    Handles sparse matrices by converting to dense if needed.
    """
    if hasattr(X_transformed, "toarray"):
        X_dense = X_transformed.toarray()
    else:
        X_dense = X_transformed

    df_X = pd.DataFrame(X_dense, columns=feature_names)
    df_y = y.reset_index(drop=True)
    df_X[TARGET_COL] = df_y
    return df_X


def write_report(
    train_df_before_smote: pd.DataFrame,
    train_df_after_smote: pd.DataFrame,
    val_df_processed: pd.DataFrame,
    test_df_processed: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
):
    """Generate a human-readable preprocessing summary report."""
    def class_distribution(df: pd.DataFrame) -> str:
        counts = df[TARGET_COL].value_counts().to_dict()
        total = len(df)
        parts = []
        for label, count in counts.items():
            perc = 100.0 * count / total
            parts.append(f"{label}: {count} ({perc:.2f}%)")
        return ", ".join(parts)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    report = []
    report.append("# Preprocessing Summary\n")
    report.append("## Dataset Sizes (after preprocessing)\n")
    report.append(f"- Train (before SMOTE): {len(train_df_before_smote)} rows")
    report.append(f"- Train (after SMOTE): {len(train_df_after_smote)} rows")
    report.append(f"- Validation: {len(val_df_processed)} rows")
    report.append(f"- Test: {len(test_df_processed)} rows\n")

    report.append("## Class Distribution\n")
    report.append(f"- Train before SMOTE: {class_distribution(train_df_before_smote)}")
    report.append(f"- Train after SMOTE: {class_distribution(train_df_after_smote)}")
    report.append(f"- Validation: {class_distribution(val_df_processed)}")
    report.append(f"- Test: {class_distribution(test_df_processed)}\n")

    report.append("## Feature Groups\n")
    report.append(f"- Numerical features ({len(num_cols)}): {', '.join(num_cols)}")
    report.append(f"- Categorical features ({len(cat_cols)}): {', '.join(cat_cols)}\n")

    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")


# ==== MAIN ORCHESTRATION ======================================================

def main():
    # Ensure directories exist
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load enriched dataset
    df = load_enriched_dataset(sample_n=None)

    # 2. Time-based splits
    train_df_raw, val_df_raw, test_df_raw = make_time_based_splits(df)

    # Save raw (unprocessed) splits for reproducibility/debugging
    train_df_raw.to_csv(TRAIN_RAW_PATH, index=False)
    val_df_raw.to_csv(VAL_RAW_PATH, index=False)
    test_df_raw.to_csv(TEST_RAW_PATH, index=False)

    # 3. Identify feature columns
    cat_cols, num_cols = identify_feature_columns(train_df_raw)

    # 4. Separate features and target
    feature_cols = num_cols + cat_cols

    X_train_raw = train_df_raw[feature_cols]
    y_train = train_df_raw[TARGET_COL]

    X_val_raw = val_df_raw[feature_cols]
    y_val = val_df_raw[TARGET_COL]

    X_test_raw = test_df_raw[feature_cols]
    y_test = test_df_raw[TARGET_COL]

    # 5. Build and fit preprocessing pipeline on TRAIN ONLY
    preprocessor = build_preprocess_pipeline(
        categorical_features=cat_cols,
        numerical_features=num_cols,
    )

    X_train_processed = preprocessor.fit_transform(X_train_raw)
    X_val_processed = preprocessor.transform(X_val_raw)
    X_test_processed = preprocessor.transform(X_test_raw)

    # Get feature names after one-hot encoding etc.
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except AttributeError:
        # Fallback if sklearn version is old
        feature_names = [f"feature_{i}" for i in range(X_train_processed.shape[1])]

    # 6. Turn into DataFrames
    train_df_processed_before_smote = dataframe_from_transformed(
        X_train_processed, y_train, feature_names
    )
    val_df_processed = dataframe_from_transformed(
        X_val_processed, y_val, feature_names
    )
    test_df_processed = dataframe_from_transformed(
        X_test_processed, y_test, feature_names
    )


    # 7. Save NO-SMOTE processed splits (needed for Autoencoder)
    train_df_processed_before_smote.to_csv(TRAIN_NOSMOTE_PATH, index=False)
    val_df_processed.to_csv(VAL_NOSMOTE_PATH, index=False)
    test_df_processed.to_csv(TEST_NOSMOTE_PATH, index=False)

# 8. Optionally apply SMOTE on TRAIN ONLY (for supervised models like XGBoost)
    if USE_SMOTE:
        sm = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = sm.fit_resample(
            train_df_processed_before_smote.drop(columns=[TARGET_COL]),
            train_df_processed_before_smote[TARGET_COL],
        )
        print("SMOTE y distribution:", pd.Series(y_train_resampled).value_counts().to_dict())

        train_df_processed_final = pd.DataFrame(X_train_resampled, columns=feature_names)
        train_df_processed_final[TARGET_COL] = y_train_resampled    
    else:
        train_df_processed_final = train_df_processed_before_smote

# 9. Persist processed splits (used by supervised training pipelines)
    train_df_processed_final.to_csv(TRAIN_PATH, index=False)
    val_df_processed.to_csv(VAL_PATH, index=False)
    test_df_processed.to_csv(TEST_PATH, index=False)


    # 10. Persist preprocessing pipeline artifact
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    # 11. Persist feature metadata
    features_metadata = {
        "target_column": TARGET_COL,
        "time_column": TIME_COL,
        "categorical_features": cat_cols,
        "numerical_features": num_cols,
        "feature_names_after_preprocessing": feature_names,

        "train_rows_before_smote": int(len(train_df_processed_before_smote)),
        "val_rows": int(len(val_df_processed)),
        "test_rows": int(len(test_df_processed)),

        "use_smote": USE_SMOTE,
        "train_nosmote_path": str(TRAIN_NOSMOTE_PATH),
        "val_nosmote_path": str(VAL_NOSMOTE_PATH),
        "test_nosmote_path": str(TEST_NOSMOTE_PATH),
        "train_rows_after_smote": int(len(train_df_processed_final)),

    }


    FEATURES_META_PATH.write_text(
        json.dumps(features_metadata, indent=2),
        encoding="utf-8",
    )

    # 12. Write human-readable report
    write_report(
        train_df_before_smote=train_df_processed_before_smote,
        train_df_after_smote=train_df_processed_final,
        val_df_processed=val_df_processed,
        test_df_processed=test_df_processed,
        cat_cols=cat_cols,
        num_cols=num_cols,
    )

    print("✅ Preprocessing finished successfully.")
    print(f"  Train processed: {TRAIN_PATH}")
    print(f"  Val processed:   {VAL_PATH}")
    print(f"  Test processed:  {TEST_PATH}")
    print(f"  Preprocessor:    {PREPROCESSOR_PATH}")
    print(f"  Features meta:   {FEATURES_META_PATH}")
    print(f"  Report:          {REPORT_PATH}")


if __name__ == "__main__":
    main()
