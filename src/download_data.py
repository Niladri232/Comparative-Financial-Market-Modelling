import yfinance as yf
import pandas as pd
import os

# ============================================================
# CONFIGURATION
# ============================================================

START_DATE = "2010-01-01"
END_DATE = "2026-09-15"

OUTPUT_DIR = "data/raw"

# Assets
ASSETS = {
    "nifty50": "^NSEI",
    "banknifty": "^NSEBANK",
    "niftyit": "^CNXIT",
    "niftypharma": "^CNXPHARMA",
    "reliance": "RELIANCE.NS",
    "tcs": "TCS.NS",
    "infosys": "INFY.NS",
    "hdfcbank": "HDFCBANK.NS",
    "icicibank": "ICICIBANK.NS",
    "itc": "ITC.NS"
}

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DOWNLOAD
# ============================================================

print("=" * 70)
print("FINANCIAL DATA DOWNLOAD")
print("=" * 70)
print(f"Period: {START_DATE} to {END_DATE}")
print(f"Output: {OUTPUT_DIR}")
print("=" * 70)


for name, ticker in ASSETS.items():

    print(f"\nDownloading: {name}")
    print(f"Ticker: {ticker}")

    try:

        data = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=False,
            progress=False
        )

        # ----------------------------------------------------
        # Check whether data was downloaded
        # ----------------------------------------------------

        if data.empty:
            print(f"WARNING: No data found for {name}")
            continue

        # ----------------------------------------------------
        # Handle Yahoo Finance multi-level columns
        # ----------------------------------------------------

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # ----------------------------------------------------
        # Reset Date from index
        # ----------------------------------------------------

        data.reset_index(inplace=True)

        # ----------------------------------------------------
        # Rename columns
        # ----------------------------------------------------

        data.rename(
            columns={
                "Adj Close": "Adj_Close"
            },
            inplace=True
        )

        # ----------------------------------------------------
        # Keep only required columns
        # ----------------------------------------------------

        required_columns = [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Adj_Close",
            "Volume"
        ]

        data = data[
            [col for col in required_columns if col in data.columns]
        ]

        # ----------------------------------------------------
        # Save CSV
        # ----------------------------------------------------

        output_file = os.path.join(
            OUTPUT_DIR,
            f"{name}.csv"
        )

        data.to_csv(
            output_file,
            index=False
        )

        # ----------------------------------------------------
        # Print summary
        # ----------------------------------------------------

        print(f"SUCCESS")
        print(f"Rows: {len(data):,}")
        print(f"Start: {data['Date'].iloc[0]}")
        print(f"End:   {data['Date'].iloc[-1]}")
        print(f"Saved: {output_file}")

    except Exception as e:

        print(f"ERROR downloading {name}")
        print(f"Reason: {e}")


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("DOWNLOAD PROCESS COMPLETED")
print("=" * 70)

print("\nFiles currently in data/raw:")

for file in os.listdir(OUTPUT_DIR):
    print(" -", file)

print("=" * 70)