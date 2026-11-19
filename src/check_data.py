import os
import pandas as pd

RAW_DIR = os.path.join("data", "raw")

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


def parse_dates(date_series):
    """
    Parse dates robustly.

    The downloaded CSV files contain two main date formats:
        YYYY-MM-DD
        DD-MM-YYYY

    We first try YYYY-MM-DD, then DD-MM-YYYY for
    values that could not be parsed.
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


def check_file(filepath):
    df = pd.read_csv(filepath)

    print("=" * 75)
    print(f"FILE: {os.path.basename(filepath)}")
    print("=" * 75)

    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # ---------------------------------------------------------
    # DATE CHECK
    # ---------------------------------------------------------
    if "Date" in df.columns:

        df["Date"] = parse_dates(df["Date"])

        print(f"First date: {df['Date'].min()}")
        print(f"Last date : {df['Date'].max()}")

        invalid_dates = df["Date"].isna().sum()
        duplicate_dates = df["Date"].duplicated().sum()

        print(f"Invalid dates   : {invalid_dates}")
        print(f"Duplicate dates : {duplicate_dates}")

    # ---------------------------------------------------------
    # MISSING VALUES
    # ---------------------------------------------------------
    print("\nMissing values:")
    print(df.isnull().sum())

    # ---------------------------------------------------------
    # DUPLICATE ROWS
    # ---------------------------------------------------------
    duplicate_rows = df.duplicated().sum()

    print(f"\nDuplicate rows: {duplicate_rows}")

    # ---------------------------------------------------------
    # BASIC STATISTICS
    # ---------------------------------------------------------
    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj_Close",
        "Volume",
    ]

    available_numeric = [
        col for col in numeric_columns
        if col in df.columns
    ]

    print("\nBasic statistics:")

    if available_numeric:
        print(
            df[available_numeric]
            .describe()
            .round(4)
        )

    # ---------------------------------------------------------
    # NON-POSITIVE PRICES
    # ---------------------------------------------------------
    price_columns = [
        col
        for col in [
            "Open",
            "High",
            "Low",
            "Close",
            "Adj_Close"
        ]
        if col in df.columns
    ]

    print("\nNon-positive price observations:")

    for col in price_columns:
        count = (df[col] <= 0).sum()
        print(f"{col}: {count}")

    print()


def main():

    if not os.path.exists(RAW_DIR):
        print(f"ERROR: Directory not found: {RAW_DIR}")
        return

    print("\nFINANCIAL DATA QUALITY CHECK")
    print("=" * 75)
    print(f"Raw data directory: {RAW_DIR}")
    print()

    missing_files = []

    for filename in FILES:

        filepath = os.path.join(
            RAW_DIR,
            filename
        )

        if not os.path.exists(filepath):
            missing_files.append(filename)
            continue

        check_file(filepath)

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------
    print("=" * 75)
    print("SUMMARY")
    print("=" * 75)

    if missing_files:

        print("Missing files:")

        for filename in missing_files:
            print(f"  - {filename}")

    else:
        print("All expected files are present.")

    print("\nData-quality check completed.")


if __name__ == "__main__":
    main()