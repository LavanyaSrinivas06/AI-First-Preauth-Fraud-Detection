from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


# ---- PATHS -------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data" / "processed"
DOCS_DIR = ROOT_DIR / "docs"

ENRICHED_PATH = DATA_DIR / "enriched.csv"
SCHEMA_PATH = DOCS_DIR / "schema_enriched.json"

VALIDATION_REPORT_PATH = DOCS_DIR / "data_validation_report.md"


# ---- HELPER FUNCTIONS --------------------------------------------------

def load_schema() -> list[dict]:
    """
    Load the expected schema from docs/schema_enriched.json.

    The schema file is a list of objects like:
    [
      {"field": "Time", "dtype": "float", "description": "..."},
      ...
    ]
    """
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema file not found at: {SCHEMA_PATH}")

    with SCHEMA_PATH.open("r", encoding="utf-8") as f:
        schema = json.load(f)

    # Expect a list of dicts with "field" and "dtype"
    if not isinstance(schema, list):
        raise ValueError("Schema file must be a list of field definitions.")

    return schema

def schema_to_required_columns(schema: list[dict]) -> dict[str, str]:
    """
    Convert the list-based schema into a dict mapping:
        column_name -> expected_pandas_dtype

    We map your logical dtypes to pandas dtype strings:
      - "float"  -> "float64"
      - "int"    -> "int64"
      - "string" -> "object"
      - "bool"   -> "bool"
    """
    dtype_map = {
        "float": "float64",
        "int": "int64",
        "string": "object",
        "bool": "bool",
    }

    required = {}
    for field_def in schema:
        col = field_def.get("field")
        logical_dtype = field_def.get("dtype")

        if col is None or logical_dtype is None:
            continue

        expected = dtype_map.get(logical_dtype, None)
        if expected is None:
            # Fallback: use logical dtype as-is if not mapped
            expected = logical_dtype

        required[col] = expected

    return required



def load_data() -> pd.DataFrame:
    """Load the processed enriched dataset."""
    if not ENRICHED_PATH.exists():
        raise FileNotFoundError(f"Enriched dataset not found at: {ENRICHED_PATH}")

    df = pd.read_csv(ENRICHED_PATH)
    return df


def check_required_columns(df: pd.DataFrame, schema: list[dict], results: list[str]) -> bool:
    """Check that all required columns exist in the dataframe."""
    required_cols = schema_to_required_columns(schema)
    missing = [col for col in required_cols.keys() if col not in df.columns]

    if missing:
        results.append(f"❌ Missing required columns: {', '.join(missing)}")
        return False

    results.append("✅ All required columns are present.")
    return True


def check_dtypes(df: pd.DataFrame, schema: list[dict], results: list[str]) -> bool:
    """
    Check that dtypes roughly match the schema.

    We compare pandas dtype name with expected mapped dtype name.
    """
    ok = True
    required_cols = schema_to_required_columns(schema)

    for col, expected_dtype in required_cols.items():
        if col not in df.columns:
            # Column presence is handled by check_required_columns
            continue

        actual = str(df[col].dtype)
        # Allow float64 for int64 columns if they might contain NaN
        if expected_dtype == "int64" and actual in ["float64", "int64"]:
            continue
        if actual != expected_dtype:
            results.append(
                f"❌ Column '{col}' has dtype {actual}, expected {expected_dtype}"
            )
            ok = False

    if ok:
        results.append("✅ All required column dtypes match the schema (basic check).")

    return ok


def check_missing_values(df: pd.DataFrame, results: list[str]) -> bool:
    """Check for missing values in important columns."""
    ok = True

    important_cols = [
        "Amount",
        "ip_reputation",
        "account_age_days",
        "token_age_days",
        "geo_distance_km",
        "device_id",
        "Class",
    ]

    for col in important_cols:
        if col not in df.columns:
            continue  # column presence is checked elsewhere
        missing = df[col].isna().sum()
        if missing > 0:
            results.append(f"❌ Column '{col}' has {missing} missing values.")
            ok = False

    if ok:
        results.append("✅ No missing values in key columns (Amount, ip_reputation, etc.).")

    return ok


def check_value_ranges(df: pd.DataFrame, results: list[str]) -> bool:
    """
    Check value ranges for key numeric columns.
    These are simple sanity checks based on domain knowledge.
    """
    ok = True

    def col_ok(series: pd.Series, description: str) -> bool:
        nonlocal ok
    # Remove NaNs (handled separately in missingness checks)
        non_null = series.dropna()

        if not non_null.empty and not non_null.all():
            results.append(f"❌ Range check failed: {description}")
            ok = False
            return False
        return True

    # Amount: >= 0
    if "Amount" in df.columns:
        col_ok(df["Amount"] >= 0, "Amount must be >= 0")

    # ip_reputation: [0, 1]
    if "ip_reputation" in df.columns:
        col_ok(df["ip_reputation"].between(0, 1), "ip_reputation must be between 0 and 1")

    # account_age_days, token_age_days: >= 0
    if "account_age_days" in df.columns:
        col_ok(df["account_age_days"] >= 0, "account_age_days must be >= 0")

    if "token_age_days" in df.columns:
        col_ok(df["token_age_days"] >= 0, "token_age_days must be >= 0")

    # geo_distance_km: >= 0
    if "geo_distance_km" in df.columns:
        col_ok(df["geo_distance_km"] >= 0, "geo_distance_km must be >= 0")

    # Class: {0,1}
    if "Class" in df.columns:
        allowed = {0, 1}
        unique_values = set(df["Class"].dropna().unique().tolist())
        if not unique_values.issubset(allowed):
            results.append(
                f"❌ Class column has invalid values: {unique_values} (expected only 0 or 1)"
            )
            ok = False
        else:
            results.append("✅ Class column contains only 0 and 1.")

    if ok:
        results.append("✅ All numeric range checks passed.")

    return ok


def build_missingness_summary(df: pd.DataFrame) -> str:
    """Generate a small missing values summary (markdown table)."""
    missing = df.isna().sum()
    total = len(df)

    lines = []
    lines.append("| Column | Missing Count | Missing % |")
    lines.append("|--------|---------------|-----------|")

    for col, count in missing.items():
        perc = 100.0 * count / total if total > 0 else 0.0
        lines.append(f"| {col} | {count} | {perc:.2f}% |")

    return "\n".join(lines)


def write_validation_report(
    df: pd.DataFrame,
    checks_output: list[str],
    all_checks_passed: bool,
) -> None:
    """Write a markdown report summarising validation checks."""
    VALIDATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Data Validation Report")
    lines.append("")
    lines.append(f"- Generated at: {datetime.utcnow().isoformat()}Z")
    lines.append(f"- Source file: `{ENRICHED_PATH.relative_to(ROOT_DIR)}`")
    lines.append(f"- Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    lines.append("")
    lines.append("## Validation Checks")
    lines.append("")
    for line in checks_output:
        lines.append(f"- {line}")
    lines.append("")

    lines.append("## Missing Values Summary")
    lines.append("")
    lines.append(build_missingness_summary(df))
    lines.append("")

    if all_checks_passed:
        lines.append("## Overall Status")
        lines.append("")
        lines.append("✅ All validation checks passed.")
    else:
        lines.append("## Overall Status")
        lines.append("")
        lines.append("❌ One or more validation checks failed. See details above.")

    VALIDATION_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


# ---- MAIN --------------------------------------------------------------


def main() -> None:
    try:
        schema = load_schema()
        df = load_data()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    checks_output: list[str] = []
    all_ok = True

    # Run checks
    if not check_required_columns(df, schema, checks_output):
        all_ok = False

    if not check_dtypes(df, schema, checks_output):
        all_ok = False

    if not check_missing_values(df, checks_output):
        all_ok = False

    if not check_value_ranges(df, checks_output):
        all_ok = False

    # Write report
    write_validation_report(df, checks_output, all_ok)

    # Exit code
    if all_ok:
        print("✅ Data validation passed. See docs/data_validation_report.md for details.")
        sys.exit(0)
    else:
        print("❌ Data validation FAILED. See docs/data_validation_report.md for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
