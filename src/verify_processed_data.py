import os
import pandas as pd
import numpy as np


PROCESSED_DIR = os.path.join("data", "processed")

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


def verify_file(filename):

    filepath = os.path.join(
        PROCESSED_DIR,
        filename
    )

    df = pd.read_csv(filepath)

    df["Date"] = pd.to_datetime(df["Date"])

    print("=" * 75)
    print(f"DATASET: {filename}")
    print("=" * 75)

    # --------------------------------------------------------
    # Basic structure
    # --------------------------------------------------------

    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # --------------------------------------------------------
    # Date checks
    # --------------------------------------------------------

    print("\nDate checks:")

    print(f"First date: {df['Date'].min().date()}")
    print(f"Last date : {df['Date'].max().date()}")
    print(f"Duplicate dates: {df['Date'].duplicated().sum()}")

    # --------------------------------------------------------
    # Price checks
    # --------------------------------------------------------

    print("\nPrice checks:")

    print(
        f"Non-positive Adj_Close: "
        f"{(df['Adj_Close'] <= 0).sum()}"
    )

    print(
        f"Missing Adj_Close: "
        f"{df['Adj_Close'].isna().sum()}"
    )

    # --------------------------------------------------------
    # Return checks
    # --------------------------------------------------------

    print("\nReturn checks:")

    print(
        f"Missing Simple_Return: "
        f"{df['Simple_Return'].isna().sum()}"
    )

    print(
        f"Missing Log_Return: "
        f"{df['Log_Return'].isna().sum()}"
    )

    # --------------------------------------------------------
    # Mathematical consistency check
    # --------------------------------------------------------

    expected_simple = df["Adj_Close"].pct_change()

    expected_log = np.log(
        df["Adj_Close"] /
        df["Adj_Close"].shift(1)
    )

    simple_difference = (
        df["Simple_Return"] -
        expected_simple
    ).abs().max()

    log_difference = (
        df["Log_Return"] -
        expected_log
    ).abs().max()

    print("\nReturn calculation verification:")

    print(
        f"Maximum Simple_Return difference: "
        f"{simple_difference:.12f}"
    )

    print(
        f"Maximum Log_Return difference: "
        f"{log_difference:.12f}"
    )

    # --------------------------------------------------------
    # Return statistics
    # --------------------------------------------------------

    returns = df["Log_Return"].dropna()

    print("\nLog-return statistics:")

    print(
        f"Mean       : {returns.mean():.8f}"
    )

    print(
        f"Std        : {returns.std():.8f}"
    )

    print(
        f"Minimum    : {returns.min():.8f}"
    )

    print(
        f"Maximum    : {returns.max():.8f}"
    )

    print(
        f"Observations: {len(returns)}"
    )

    print()


def main():

    print()
    print("=" * 75)
    print("PROCESSED DATA VERIFICATION")
    print("=" * 75)
    print()

    for filename in FILES:

        filepath = os.path.join(
            PROCESSED_DIR,
            filename
        )

        if not os.path.exists(filepath):

            print(
                f"ERROR: Missing file: {filename}"
            )

            continue

        verify_file(filename)

    print("=" * 75)
    print("VERIFICATION COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()