import os
import pandas as pd
import numpy as np


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")


# ============================================================
# DATASETS
# ============================================================

FILES = [
    "banknifty.csv",
    "hdfcbank.csv",
    "icicibank.csv",
    "infosys.csv",
    "itc.csv",
    "nifty50.csv",
    "niftyit.csv",
    "niftypharma.csv",
    "reliance.csv",
    "tcs.csv",
]


# ============================================================
# DATE PARSER
# ============================================================

def parse_dates(date_series):
    """
    Parse dates robustly.

    The raw datasets contain dates in either:
        YYYY-MM-DD
        DD-MM-YYYY
    """

    dates = pd.to_datetime(
        date_series,
        format="%Y-%m-%d",
        errors="coerce"
    )

    missing_mask = dates.isna()

    if missing_mask.any():
        dates.loc[missing_mask] = pd.to_datetime(
            date_series.loc[missing_mask],
            format="%d-%m-%Y",
            errors="coerce"
        )

    return dates


# ============================================================
# PROCESS ONE DATASET
# ============================================================

def preprocess_file(filename):

    input_path = os.path.join(RAW_DIR, filename)

    output_path = os.path.join(
        PROCESSED_DIR,
        filename
    )

    print("=" * 75)
    print(f"PROCESSING: {filename}")
    print("=" * 75)

    # --------------------------------------------------------
    # Load raw data
    # --------------------------------------------------------

    df = pd.read_csv(input_path)

    print(f"Raw rows: {len(df)}")

    # --------------------------------------------------------
    # Parse dates
    # --------------------------------------------------------

    df["Date"] = parse_dates(df["Date"])

    invalid_dates = df["Date"].isna().sum()

    if invalid_dates > 0:
        raise ValueError(
            f"{filename} contains {invalid_dates} invalid dates."
        )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_values("Date").reset_index(drop=True)

    # --------------------------------------------------------
    # Remove duplicate dates
    # --------------------------------------------------------

    duplicate_dates = df["Date"].duplicated().sum()

    if duplicate_dates > 0:
        print(
            f"Warning: removing {duplicate_dates} duplicate dates."
        )

        df = (
            df.drop_duplicates(
                subset="Date",
                keep="first"
            )
            .reset_index(drop=True)
        )

    # --------------------------------------------------------
    # Validate adjusted close
    # --------------------------------------------------------

    if "Adj_Close" not in df.columns:
        raise ValueError(
            f"{filename} does not contain Adj_Close."
        )

    invalid_prices = (df["Adj_Close"] <= 0).sum()

    if invalid_prices > 0:
        raise ValueError(
            f"{filename} contains "
            f"{invalid_prices} non-positive adjusted prices."
        )

    # --------------------------------------------------------
    # Calculate simple returns
    #
    # R_t = (P_t - P_{t-1}) / P_{t-1}
    # --------------------------------------------------------

    df["Simple_Return"] = (
        df["Adj_Close"].pct_change()
    )

    # --------------------------------------------------------
    # Calculate log returns
    #
    # r_t = ln(P_t / P_{t-1})
    # --------------------------------------------------------

    df["Log_Return"] = np.log(
        df["Adj_Close"] /
        df["Adj_Close"].shift(1)
    )

    # --------------------------------------------------------
    # Save processed dataset
    # --------------------------------------------------------

    df.to_csv(
        output_path,
        index=False
    )

    # --------------------------------------------------------
    # Processing summary
    # --------------------------------------------------------

    print(f"Processed rows: {len(df)}")
    print(f"First date    : {df['Date'].min().date()}")
    print(f"Last date     : {df['Date'].max().date()}")
    print(f"Duplicate dates removed: {duplicate_dates}")
    print(f"Output        : {output_path}")

    print("\nMissing values after preprocessing:")

    print(
        df[
            [
                "Date",
                "Adj_Close",
                "Simple_Return",
                "Log_Return"
            ]
        ]
        .isnull()
        .sum()
    )

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Check directories
    # --------------------------------------------------------

    if not os.path.exists(RAW_DIR):
        raise FileNotFoundError(
            f"Raw data directory not found: {RAW_DIR}"
        )

    # Create processed directory if necessary
    os.makedirs(
        PROCESSED_DIR,
        exist_ok=True
    )

    print()
    print("=" * 75)
    print("FINANCIAL DATA PREPROCESSING")
    print("=" * 75)
    print(f"Raw directory      : {RAW_DIR}")
    print(f"Processed directory: {PROCESSED_DIR}")
    print()

    # --------------------------------------------------------
    # Process all datasets
    # --------------------------------------------------------

    missing_files = []

    for filename in FILES:

        input_path = os.path.join(
            RAW_DIR,
            filename
        )

        if not os.path.exists(input_path):
            missing_files.append(filename)
            continue

        preprocess_file(filename)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("=" * 75)
    print("PREPROCESSING SUMMARY")
    print("=" * 75)

    if missing_files:

        print("Missing input files:")

        for filename in missing_files:
            print(f"  - {filename}")

    else:

        print("All 10 datasets processed successfully.")

    print()
    print(
        f"Processed files are stored in: {PROCESSED_DIR}"
    )
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
