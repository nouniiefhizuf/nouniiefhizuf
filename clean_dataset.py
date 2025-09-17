#!/usr/bin/env python3
"""
Data Cleaning Pipeline for retail transactions dataset.

This script performs the following steps:
 1) Load and explore the dataset
 2) Identify and handle missing values (drop/impute)
 3) Detect and address outliers via IQR rule (and optional boxplots)
 4) Remove duplicates
 5) Standardize and normalize numerical columns
 6) Handle inconsistent categorical and date fields
 7) Validate and save cleaned data to CSV

Usage examples:
  - python clean_dataset.py --input /workspace/Uncleaned_Data_Set.csv \
      --output /workspace/Cleaned_DataSet.csv

  - python clean_dataset.py --download-url "https://example.com/Uncleaned_Data_Set.csv" \
      --input /workspace/Uncleaned_Data_Set.csv --output /workspace/Cleaned_DataSet.csv

Notes:
 - If --download-url is provided and the input file does not exist, the file
   is downloaded to --input before processing.
 - Boxplots are saved under the directory specified by --plots-dir.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


def safe_imports_for_plots() -> None:
    """Import heavy plotting libraries lazily only when needed."""
    global plt, sns
    import matplotlib.pyplot as plt  # type: ignore
    import seaborn as sns  # type: ignore


MISSING_MARKERS = [
    "", " ", "na", "n/a", "null", "none", "unknown", "nan", "error",
    "UNKNOWN", "ERROR", "N/A", "NaN", "Null", "None"
]


@dataclass
class CleaningConfig:
    input_path: str
    output_path: str
    plots_dir: Optional[str]
    minmax_column: Optional[str]
    remove_outliers: bool
    download_url: Optional[str]
    row_missing_threshold: float


def parse_args() -> CleaningConfig:
    parser = argparse.ArgumentParser(description="Clean a retail transactions dataset")
    parser.add_argument("--input", dest="input_path", required=True, help="Path to input CSV")
    parser.add_argument("--output", dest="output_path", required=True, help="Path to output cleaned CSV")
    parser.add_argument("--plots-dir", dest="plots_dir", default="/workspace/outputs/boxplots", help="Directory to save boxplots")
    parser.add_argument("--minmax-column", dest="minmax_column", default="Total Spent", help="Column to normalize to [0,1] using MinMaxScaler")
    parser.add_argument("--no-remove-outliers", dest="no_remove_outliers", action="store_true", help="Do not remove outliers (still plots if plots-dir provided)")
    parser.add_argument("--download-url", dest="download_url", default=None, help="Optional URL to download the CSV if input path is missing")
    parser.add_argument("--row-missing-threshold", dest="row_missing_threshold", type=float, default=0.7, help="Drop rows missing more than this fraction of columns (0-1)")

    args = parser.parse_args()
    return CleaningConfig(
        input_path=args.input_path,
        output_path=args.output_path,
        plots_dir=args.plots_dir,
        minmax_column=args.minmax_column,
        remove_outliers=not args.no_remove_outliers,
        download_url=args.download_url,
        row_missing_threshold=args.row_missing_threshold,
    )


def maybe_download_input(config: CleaningConfig) -> None:
    if os.path.exists(config.input_path):
        return
    if not config.download_url:
        print(f"[BLOCKER] Input file not found: {config.input_path}. Provide the CSV or pass --download-url.")
        sys.exit(2)
    try:
        import requests  # type: ignore
    except Exception as exc:  # pragma: no cover - import error reporting
        print("Failed to import requests to download the file:", exc)
        sys.exit(2)
    print(f"Downloading dataset from {config.download_url} ...")
    resp = requests.get(config.download_url, timeout=60)
    resp.raise_for_status()
    os.makedirs(os.path.dirname(config.input_path), exist_ok=True)
    with open(config.input_path, "wb") as f:
        f.write(resp.content)
    print(f"Saved to {config.input_path}")


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {c: c.strip().replace("_", " ").replace("  ", " ") for c in df.columns}
    df = df.rename(columns=mapping)
    return df


def explore_dataset(df: pd.DataFrame) -> None:
    print("=== Head (first 5 rows) ===")
    print(df.head())
    print("\n=== Info ===")
    buf = []
    df.info(buf=buf)
    print("\n".join(str(x) for x in buf))
    print("\n=== Describe (numeric) ===")
    print(df.describe(include=[np.number]))
    print("\n=== Describe (categorical) ===")
    print(df.describe(include=[object]))


def coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce known numeric and date columns where possible."""
    candidate_numeric = [
        col for col in df.columns if any(k in col.lower() for k in ["quantity", "price", "total", "amount", "unit"])
    ]
    for col in candidate_numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Transaction date normalization
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def handle_inconsistencies(df: pd.DataFrame) -> pd.DataFrame:
    # Lowercase and strip categorical columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # Location normalization examples
    for col in df.columns:
        if "location" in col.lower():
            df[col] = (
                df[col]
                .str.replace("in store", "in-store", regex=False)
                .str.replace("instore", "in-store", regex=False)
                .str.replace("take away", "takeaway", regex=False)
            )

    # Payment method normalization examples
    for col in df.columns:
        if "payment" in col.lower():
            normalized = (
                df[col]
                .str.replace("creditcard", "credit card", regex=False)
                .str.replace("debitcard", "debit card", regex=False)
                .str.replace("digitalwallet", "digital wallet", regex=False)
            )
            df[col] = normalized

    # Replace marker strings with NaN uniformly across all columns
    df = df.replace(MISSING_MARKERS, np.nan)

    return df


def complete_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    # Attempt to compute Total Spent = Quantity * Price Per Unit if missing
    quantity_col = next((c for c in df.columns if c.lower() == "quantity"), None)
    price_col = next((c for c in df.columns if c.lower() in ("price per unit", "price", "unit price")), None)
    total_col = next((c for c in df.columns if c.lower() in ("total spent", "total", "amount")), None)

    if total_col and quantity_col and price_col:
        missing_total_mask = df[total_col].isna()
        computed_total = df.loc[missing_total_mask, quantity_col] * df.loc[missing_total_mask, price_col]
        df.loc[missing_total_mask, total_col] = computed_total

    return df


def drop_and_impute_missing(df: pd.DataFrame, row_missing_threshold: float) -> pd.DataFrame:
    # Drop rows that are mostly empty
    frac_missing_by_row = df.isna().mean(axis=1)
    df = df.loc[frac_missing_by_row <= row_missing_threshold].copy()

    # For numeric columns, fill with median
    for col in df.select_dtypes(include=[np.number]).columns:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

    # For categorical columns, fill with mode (or 'unknown' if mode missing)
    for col in df.select_dtypes(include=["object"]).columns:
        mode_series = df[col].mode(dropna=True)
        fill_value = mode_series.iloc[0] if not mode_series.empty else "unknown"
        df[col] = df[col].fillna(fill_value)

    # Dates: fill with mode if any
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.datetime64):
            mode_series = df[col].mode(dropna=True)
            if not mode_series.empty:
                df[col] = df[col].fillna(mode_series.iloc[0])

    return df


def detect_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return pd.DataFrame(index=df.index, data={"is_outlier": [False] * len(df)})

    outlier_mask = pd.Series(False, index=df.index)
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_mask = outlier_mask | (df[col] < lower_bound) | (df[col] > upper_bound)
    return pd.DataFrame(index=df.index, data={"is_outlier": outlier_mask})


def plot_boxplots(df: pd.DataFrame, plots_dir: str) -> None:
    os.makedirs(plots_dir, exist_ok=True)
    safe_imports_for_plots()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        out_path = os.path.join(plots_dir, f"boxplot_{col.replace(' ', '_')}.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().copy()
    after = len(df)
    print(f"Removed {before - after} duplicate rows. New shape: {df.shape}")
    return df


def scale_features(df: pd.DataFrame, minmax_column: Optional[str]) -> pd.DataFrame:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler  # type: ignore

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    if numeric_cols:
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(df[numeric_cols])
        scaled_df = pd.DataFrame(scaled_values, columns=[f"{c}_z" for c in numeric_cols], index=df.index)
        df = pd.concat([df, scaled_df], axis=1)
        print(f"Standardized columns: {', '.join(numeric_cols)} -> suffixed with _z")

    if minmax_column and minmax_column in df.columns:
        # If the target is not numeric yet, try to convert
        if not np.issubdtype(df[minmax_column].dtype, np.number):
            df[minmax_column] = pd.to_numeric(df[minmax_column], errors="coerce")
            df[minmax_column] = df[minmax_column].fillna(df[minmax_column].median())
        scaler_mm = MinMaxScaler()
        df[f"{minmax_column}_minmax"] = scaler_mm.fit_transform(df[[minmax_column]])
        print(f"Normalized column '{minmax_column}' to '{minmax_column}_minmax' in [0,1]")
    else:
        if minmax_column:
            print(f"[WARN] MinMax column '{minmax_column}' not found. Skipping normalization.")

    return df


def validate(df: pd.DataFrame) -> None:
    null_counts = df.isna().sum()
    any_nulls = int(null_counts.sum())
    print("=== Validation ===")
    print(f"Shape: {df.shape}")
    print("Null counts by column:\n", null_counts[null_counts > 0])
    if any_nulls == 0:
        print("No missing values remain.")
    else:
        print(f"[WARN] Remaining missing values: {any_nulls}")


def main() -> None:
    config = parse_args()

    maybe_download_input(config)

    if not os.path.exists(config.input_path):
        print(f"[BLOCKER] Input file still not found at {config.input_path}.")
        sys.exit(2)

    # Load CSV (treat common missing markers as NaN)
    df = pd.read_csv(config.input_path, na_values=MISSING_MARKERS, keep_default_na=True)
    df = standardize_column_names(df)

    print("Loaded dataset with shape:", df.shape)
    explore_dataset(df)

    # Inconsistencies and type coercion
    df = handle_inconsistencies(df)
    df = coerce_dtypes(df)
    df = complete_derived_fields(df)

    # Missing values handling
    df = drop_and_impute_missing(df, row_missing_threshold=config.row_missing_threshold)

    # Outlier detection and optional removal
    outliers = detect_outliers_iqr(df)
    num_outliers = int(outliers["is_outlier"].sum())
    print(f"Detected {num_outliers} outlier rows via IQR rule.")
    if config.plots_dir:
        try:
            plot_boxplots(df, config.plots_dir)
            print(f"Saved boxplots to: {config.plots_dir}")
        except Exception as exc:
            print(f"[WARN] Failed to generate boxplots: {exc}")
    if config.remove_outliers and num_outliers > 0:
        df = df.loc[~outliers["is_outlier"]].copy()
        print(f"Removed outliers. New shape: {df.shape}")

    # Duplicates
    df = remove_duplicates(df)

    # Scaling
    df = scale_features(df, config.minmax_column)

    # Dates to consistent format (ISO)
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.datetime64):
            df[col] = df[col].dt.strftime("%Y-%m-%d")

    # Final validation and save
    validate(df)
    os.makedirs(os.path.dirname(config.output_path) or ".", exist_ok=True)
    df.to_csv(config.output_path, index=False)
    print(f"Saved cleaned dataset to: {config.output_path}")


if __name__ == "__main__":
    main()

