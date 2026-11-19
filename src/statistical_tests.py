import os

import numpy as np
import pandas as pd

from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller


# ============================================================
# CONFIGURATION
# ============================================================

PROCESSED_DIR = os.path.join("data", "processed")
RESULTS_DIR = "results"

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

LJUNG_BOX_LAGS = [1, 5, 10, 20]
ARCH_LAGS = 10


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
# VARIANCE RATIO TEST
# ============================================================

def variance_ratio_test(returns, lag=5):
    """
    Lo-MacKinlay-style variance ratio statistic.

    Under a random walk:
        Var(r_t + ... + r_{t+k-1}) ≈ k Var(r_t)

    Therefore:

        VR(k) = Var(k-period returns)
               ----------------------
                  k Var(1-period returns)

    VR approximately equal to 1 supports the
    uncorrelated-increment assumption.
    """

    returns = np.asarray(returns)

    returns = returns[np.isfinite(returns)]

    n = len(returns)

    if n <= lag:
        return np.nan, np.nan

    # One-period variance
    var_1 = np.var(returns, ddof=1)

    # Construct overlapping k-period returns
    cumulative_returns = np.convolve(
        returns,
        np.ones(lag),
        mode="valid"
    )

    var_k = np.var(cumulative_returns, ddof=1)

    variance_ratio = var_k / (lag * var_1)

    # Approximate standard error under homoscedastic random walk
    se = np.sqrt(
        (2 * (2 * lag - 1) * (lag - 1))
        / (3 * lag * n)
    )

    if se > 0:
        z_statistic = (variance_ratio - 1) / se
    else:
        z_statistic = np.nan

    return variance_ratio, z_statistic


# ============================================================
# TEST ONE DATASET
# ============================================================

def test_dataset(filename):

    df = load_data(filename)

    asset = filename.replace(".csv", "")

    returns = df["Log_Return"].dropna()

    prices = df["Adj_Close"].dropna()

    # --------------------------------------------------------
    # Ljung-Box test
    # --------------------------------------------------------

    ljung_box = acorr_ljungbox(
        returns,
        lags=LJUNG_BOX_LAGS,
        return_df=True
    )

    lb_pvalue_10 = ljung_box.loc[
        10,
        "lb_pvalue"
    ]

    # --------------------------------------------------------
    # ADF test on price
    # --------------------------------------------------------

    adf_price = adfuller(
        prices,
        autolag="AIC"
    )

    adf_price_statistic = adf_price[0]
    adf_price_pvalue = adf_price[1]

    # --------------------------------------------------------
    # ADF test on returns
    # --------------------------------------------------------

    adf_returns = adfuller(
        returns,
        autolag="AIC"
    )

    adf_returns_statistic = adf_returns[0]
    adf_returns_pvalue = adf_returns[1]

    # --------------------------------------------------------
    # ARCH test
    # --------------------------------------------------------

    arch_test = het_arch(
        returns,
        nlags=ARCH_LAGS
    )

    arch_lm_statistic = arch_test[0]
    arch_lm_pvalue = arch_test[1]

    # --------------------------------------------------------
    # Variance Ratio
    # --------------------------------------------------------

    vr_2, vr_z_2 = variance_ratio_test(
        returns,
        lag=2
    )

    vr_5, vr_z_5 = variance_ratio_test(
        returns,
        lag=5
    )

    vr_10, vr_z_10 = variance_ratio_test(
        returns,
        lag=10
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    result = {
        "Asset": asset,

        "LjungBox_pvalue_lag1":
            ljung_box.loc[1, "lb_pvalue"],

        "LjungBox_pvalue_lag5":
            ljung_box.loc[5, "lb_pvalue"],

        "LjungBox_pvalue_lag10":
            ljung_box.loc[10, "lb_pvalue"],

        "LjungBox_pvalue_lag20":
            ljung_box.loc[20, "lb_pvalue"],

        "ADF_Price_Statistic":
            adf_price_statistic,

        "ADF_Price_pvalue":
            adf_price_pvalue,

        "ADF_Return_Statistic":
            adf_returns_statistic,

        "ADF_Return_pvalue":
            adf_returns_pvalue,

        "ARCH_LM_Statistic":
            arch_lm_statistic,

        "ARCH_LM_pvalue":
            arch_lm_pvalue,

        "Variance_Ratio_2":
            vr_2,

        "Variance_Ratio_Z_2":
            vr_z_2,

        "Variance_Ratio_5":
            vr_5,

        "Variance_Ratio_Z_5":
            vr_z_5,

        "Variance_Ratio_10":
            vr_10,

        "Variance_Ratio_Z_10":
            vr_z_10,
    }

    return result, ljung_box


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print()
    print("=" * 75)
    print("STATISTICAL VALIDATION OF MARKET RETURNS")
    print("=" * 75)
    print()

    all_results = []

    for filename in FILES:

        filepath = os.path.join(
            PROCESSED_DIR,
            filename
        )

        if not os.path.exists(filepath):
            print(f"ERROR: Missing file: {filename}")
            continue

        print("=" * 75)
        print(f"TESTING: {filename}")
        print("=" * 75)

        result, ljung_box = test_dataset(filename)

        all_results.append(result)

        print()
        print("Ljung-Box test:")

        for lag in LJUNG_BOX_LAGS:

            pvalue = ljung_box.loc[
                lag,
                "lb_pvalue"
            ]

            print(
                f"  Lag {lag:2d} p-value: "
                f"{pvalue:.6e}"
            )

        print()
        print("ADF test on price:")

        print(
            f"  Statistic: "
            f"{result['ADF_Price_Statistic']:.6f}"
        )

        print(
            f"  p-value  : "
            f"{result['ADF_Price_pvalue']:.6e}"
        )

        print()
        print("ADF test on log returns:")

        print(
            f"  Statistic: "
            f"{result['ADF_Return_Statistic']:.6f}"
        )

        print(
            f"  p-value  : "
            f"{result['ADF_Return_pvalue']:.6e}"
        )

        print()
        print("ARCH test:")

        print(
            f"  LM statistic: "
            f"{result['ARCH_LM_Statistic']:.6f}"
        )

        print(
            f"  p-value     : "
            f"{result['ARCH_LM_pvalue']:.6e}"
        )

        print()
        print("Variance Ratio tests:")

        for lag in [2, 5, 10]:

            vr = result[
                f"Variance_Ratio_{lag}"
            ]

            z = result[
                f"Variance_Ratio_Z_{lag}"
            ]

            print(
                f"  Lag {lag:2d}: "
                f"VR = {vr:.6f}, "
                f"Z = {z:.6f}"
            )

        print()

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(all_results)

    output_path = os.path.join(
        RESULTS_DIR,
        "statistical_tests.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print("=" * 75)
    print("STATISTICAL TESTING COMPLETED")
    print("=" * 75)
    print()
    print(f"Results saved to: {output_path}")
    print()
    print(f"Assets tested: {len(results_df)}")
    print()


if __name__ == "__main__":
    main()