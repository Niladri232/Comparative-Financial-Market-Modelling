import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import skew, kurtosis, jarque_bera
from statsmodels.graphics.tsaplots import plot_acf


# ============================================================
# CONFIGURATION
# ============================================================

PROCESSED_DIR = os.path.join("data", "processed")
RESULTS_DIR = "results"
FIGURES_DIR = "figures"

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

TRADING_DAYS = 252


# ============================================================
# DIRECTORIES
# ============================================================

def create_directories():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_data(filename):
    filepath = os.path.join(PROCESSED_DIR, filename)

    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date").reset_index(drop=True)

    return df


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

def calculate_statistics(df, filename):
    returns = df["Log_Return"].dropna()

    mean_daily = returns.mean()
    std_daily = returns.std()

    annualized_return = mean_daily * TRADING_DAYS
    annualized_volatility = std_daily * np.sqrt(TRADING_DAYS)

    jb_result = jarque_bera(returns)

    statistics = {
        "Dataset": filename.replace(".csv", ""),
        "Observations": len(returns),
        "Start_Date": df["Date"].min().date(),
        "End_Date": df["Date"].max().date(),

        "Initial_Price": df["Adj_Close"].iloc[0],
        "Final_Price": df["Adj_Close"].iloc[-1],

        "Mean_Daily_Log_Return": mean_daily,
        "Daily_Volatility": std_daily,

        "Annualized_Return": annualized_return,
        "Annualized_Volatility": annualized_volatility,

        "Minimum_Return": returns.min(),
        "Maximum_Return": returns.max(),

        "Skewness": skew(returns),
        "Excess_Kurtosis": kurtosis(returns, fisher=True),

        "Jarque_Bera_Statistic": jb_result.statistic,
        "Jarque_Bera_pvalue": jb_result.pvalue,

        "5th_Percentile": returns.quantile(0.05),
        "1st_Percentile": returns.quantile(0.01),

        "95th_Percentile": returns.quantile(0.95),
        "99th_Percentile": returns.quantile(0.99),
    }

    return statistics


# ============================================================
# PRICE PLOT
# ============================================================

def create_price_plot(df, filename):
    asset_name = filename.replace(".csv", "").upper()

    plt.figure(figsize=(12, 6))

    plt.plot(
        df["Date"],
        df["Adj_Close"],
        linewidth=1
    )

    plt.title(f"{asset_name} Adjusted Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Adjusted Close")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        f"{asset_name.lower()}_price.png"
    )

    plt.savefig(output_path, dpi=150)
    plt.close()


# ============================================================
# RETURN DISTRIBUTION
# ============================================================

def create_return_distribution(df, filename):
    asset_name = filename.replace(".csv", "").upper()

    returns = df["Log_Return"].dropna()

    plt.figure(figsize=(10, 6))

    plt.hist(
        returns,
        bins=100,
        density=True
    )

    plt.title(f"{asset_name} Log-Return Distribution")
    plt.xlabel("Log Return")
    plt.ylabel("Density")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        f"{asset_name.lower()}_return_distribution.png"
    )

    plt.savefig(output_path, dpi=150)
    plt.close()


# ============================================================
# RETURN TIME SERIES
# ============================================================

def create_return_plot(df, filename):
    asset_name = filename.replace(".csv", "").upper()

    plt.figure(figsize=(12, 5))

    plt.plot(
        df["Date"],
        df["Log_Return"],
        linewidth=0.7
    )

    plt.axhline(
        y=0,
        linewidth=1
    )

    plt.title(f"{asset_name} Daily Log Returns")
    plt.xlabel("Date")
    plt.ylabel("Log Return")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        f"{asset_name.lower()}_returns.png"
    )

    plt.savefig(output_path, dpi=150)
    plt.close()


# ============================================================
# ACF OF RETURNS
# ============================================================

def create_return_acf(df, filename):
    asset_name = filename.replace(".csv", "").upper()

    returns = df["Log_Return"].dropna()

    fig, ax = plt.subplots(figsize=(10, 5))

    plot_acf(
        returns,
        lags=30,
        alpha=0.05,
        ax=ax
    )

    ax.set_title(f"{asset_name} ACF of Log Returns")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")

    fig.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        f"{asset_name.lower()}_return_acf.png"
    )

    fig.savefig(output_path, dpi=150)
    plt.close(fig)

# ============================================================
# ACF OF SQUARED RETURNS
# ============================================================

def create_squared_return_acf(df, filename):
    asset_name = filename.replace(".csv", "").upper()

    returns = df["Log_Return"].dropna()

    squared_returns = returns ** 2

    fig, ax = plt.subplots(figsize=(10, 5))

    plot_acf(
        squared_returns,
        lags=30,
        alpha=0.05,
        ax=ax
    )

    ax.set_title(f"{asset_name} ACF of Squared Log Returns")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")

    fig.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        f"{asset_name.lower()}_squared_return_acf.png"
    )

    fig.savefig(output_path, dpi=150)
    plt.close(fig)

# ============================================================
# MAIN
# ============================================================

def main():

    create_directories()

    print()
    print("=" * 75)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 75)
    print()

    all_statistics = []

    for filename in FILES:

        print("=" * 75)
        print(f"ANALYZING: {filename}")
        print("=" * 75)

        filepath = os.path.join(PROCESSED_DIR, filename)

        if not os.path.exists(filepath):
            print(f"ERROR: File not found: {filepath}")
            print()
            continue

        df = load_data(filename)

        statistics = calculate_statistics(df, filename)
        all_statistics.append(statistics)

        print(f"Observations        : {statistics['Observations']}")
        print(f"Start date          : {statistics['Start_Date']}")
        print(f"End date            : {statistics['End_Date']}")

        print()
        print("Price:")
        print(f"Initial price       : {statistics['Initial_Price']:.4f}")
        print(f"Final price         : {statistics['Final_Price']:.4f}")

        print()
        print("Log returns:")
        print(
            f"Mean daily return   : "
            f"{statistics['Mean_Daily_Log_Return']:.8f}"
        )

        print(
            f"Daily volatility    : "
            f"{statistics['Daily_Volatility']:.8f}"
        )

        print(
            f"Annualized return   : "
            f"{statistics['Annualized_Return']:.6f}"
        )

        print(
            f"Annualized volatility: "
            f"{statistics['Annualized_Volatility']:.6f}"
        )

        print()
        print("Distribution:")
        print(
            f"Minimum return     : "
            f"{statistics['Minimum_Return']:.8f}"
        )

        print(
            f"Maximum return     : "
            f"{statistics['Maximum_Return']:.8f}"
        )

        print(
            f"Skewness           : "
            f"{statistics['Skewness']:.6f}"
        )

        print(
            f"Excess kurtosis    : "
            f"{statistics['Excess_Kurtosis']:.6f}"
        )

        print()
        print("Jarque-Bera normality test:")
        print(
            f"Statistic          : "
            f"{statistics['Jarque_Bera_Statistic']:.4f}"
        )

        print(
            f"p-value            : "
            f"{statistics['Jarque_Bera_pvalue']:.6e}"
        )

        print()

        # Generate figures
        create_price_plot(df, filename)
        create_return_plot(df, filename)
        create_return_distribution(df, filename)
        create_return_acf(df, filename)
        create_squared_return_acf(df, filename)

        print("Figures generated.")
        print()

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary_df = pd.DataFrame(all_statistics)

    summary_path = os.path.join(
        RESULTS_DIR,
        "eda_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print("=" * 75)
    print("EDA COMPLETED")
    print("=" * 75)
    print()
    print(f"Summary saved to: {summary_path}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print()
    print("Assets analyzed:", len(summary_df))
    print()


if __name__ == "__main__":
    main()