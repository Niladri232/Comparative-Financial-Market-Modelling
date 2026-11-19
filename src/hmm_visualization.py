
"""
Experiment 5: HMM Market Regime Visualisation

Creates:
1. NIFTY 50 price with 2-state regimes
2. NIFTY 50 price with 3-state regimes
3. 2-state regime timeline
4. 3-state regime timeline
5. 3-state posterior probability plot
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PRICE_FILE = BASE_DIR / "data" / "processed" / "nifty50.csv"

RESULTS_DIR = BASE_DIR / "results" / "hmm"
FIGURES_DIR = BASE_DIR / "figures" / "hmm"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (16, 7),
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 120
})

COLORS_2 = {
    "Low Volatility": "seagreen",
    "High Volatility": "crimson"
}

COLORS_3 = {
    "Low Volatility": "seagreen",
    "Medium Volatility": "goldenrod",
    "High Volatility": "crimson"
}


# ============================================================
# 2. LOAD PRICE DATA
# ============================================================

prices = pd.read_csv(PRICE_FILE)

prices["Date"] = pd.to_datetime(prices["Date"])

prices = prices[["Date", "Adj_Close"]].copy()

prices = prices.dropna()

prices = prices.sort_values("Date").reset_index(drop=True)


def load_classification(n_states):

    file_path = (
        RESULTS_DIR
        / f"hmm_{n_states}_state_classification.csv"
    )

    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(df["Date"])

    return df


regime_2 = load_classification(2)

regime_3 = load_classification(3)

# Merge price and regime information by date.
data_2 = prices.merge(
    regime_2[["Date", "State", "Regime"]],
    on="Date",
    how="inner"
)

data_3 = prices.merge(
    regime_3[["Date", "State", "Regime"]],
    on="Date",
    how="inner"
)

print("\n========== VISUALISATION DATA ==========")

print("Price observations:", len(prices))
print("Two-state observations:", len(data_2))
print("Three-state observations:", len(data_3))

print("Price range:",
      prices["Date"].min().date(),
      "to",
      prices["Date"].max().date())


# ============================================================
# 3. FUNCTION: PRICE WITH REGIME CLASSIFICATION
# ============================================================

def plot_price_regimes(data, n_states):

    if n_states == 2:
        color_map = COLORS_2
        title = "NIFTY 50 Price and Two-State HMM Regimes"
        filename = "hmm_2_state_price_regimes.png"

    else:
        color_map = COLORS_3
        title = "NIFTY 50 Price and Three-State HMM Regimes"
        filename = "hmm_3_state_price_regimes.png"

    fig, ax = plt.subplots(figsize=(16, 7))

    ax.plot(
        data["Date"],
        data["Adj_Close"],
        color="lightgray",
        linewidth=1.0,
        label="NIFTY 50 Adjusted Close",
        zorder=1
    )

    for regime, color in color_map.items():

        subset = data[data["Regime"] == regime]

        ax.scatter(
            subset["Date"],
            subset["Adj_Close"],
            color=color,
            s=7,
            alpha=0.75,
            label=regime,
            zorder=2
        )

    ax.set_title(title)

    ax.set_xlabel("Date")
    ax.set_ylabel("NIFTY 50 Adjusted Close")

    ax.legend(loc="upper left")

    ax.grid(alpha=0.25)

    ax.xaxis.set_major_locator(
        mdates.YearLocator(2)
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y")
    )

    plt.xticks(rotation=0)

    plt.tight_layout()

    output = FIGURES_DIR / filename

    plt.savefig(output, dpi=300, bbox_inches="tight")

    plt.close()

    print("Saved:", filename)


# ============================================================
# 4. FUNCTION: REGIME TIMELINE
# ============================================================

def plot_regime_timeline(data, n_states):

    if n_states == 2:
        color_map = COLORS_2
        regime_order = [
            "Low Volatility",
            "High Volatility"
        ]
        title = "Two-State HMM: Market Regime Timeline"
        filename = "hmm_2_state_timeline.png"

    else:
        color_map = COLORS_3
        regime_order = [
            "Low Volatility",
            "Medium Volatility",
            "High Volatility"
        ]
        title = "Three-State HMM: Market Regime Timeline"
        filename = "hmm_3_state_timeline.png"

    fig, ax = plt.subplots(figsize=(16, 5))

    for regime in regime_order:

        subset = data[data["Regime"] == regime]

        ax.scatter(
            subset["Date"],
            subset["State"],
            color=color_map[regime],
            s=7,
            alpha=0.75,
            label=regime
        )

    ax.set_yticks(range(n_states))
    ax.set_yticklabels(regime_order)

    ax.set_title(title)

    ax.set_xlabel("Date")
    ax.set_ylabel("Estimated Market Regime")

    ax.legend(loc="upper left")

    ax.grid(alpha=0.25)

    ax.xaxis.set_major_locator(
        mdates.YearLocator(2)
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y")
    )

    plt.tight_layout()

    output = FIGURES_DIR / filename

    plt.savefig(output, dpi=300, bbox_inches="tight")

    plt.close()

    print("Saved:", filename)


# ============================================================
# 5. THREE-STATE POSTERIOR PROBABILITIES
# ============================================================

def plot_posterior_probabilities(data):

    probability_columns = [
        "Probability_Low_Volatility",
        "Probability_Medium_Volatility",
        "Probability_High_Volatility"
    ]

    regime_names = [
        "Low Volatility",
        "Medium Volatility",
        "High Volatility"
    ]

    colors = [
        COLORS_3[name] for name in regime_names
    ]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16, 10),
        sharex=True
    )

    for i, ax in enumerate(axes):

        ax.plot(
            data["Date"],
            data[probability_columns[i]],
            color=colors[i],
            linewidth=0.8
        )

        ax.set_ylabel("Probability")

        ax.set_ylim(-0.02, 1.02)

        ax.set_title(
            f"Posterior Probability: {regime_names[i]}"
        )

        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Date")

    axes[-1].xaxis.set_major_locator(
        mdates.YearLocator(2)
    )

    axes[-1].xaxis.set_major_formatter(
        mdates.DateFormatter("%Y")
    )

    fig.suptitle(
        "Three-State HMM: Estimated Market Regime Probabilities",
        fontsize=16,
        y=1.01
    )

    plt.tight_layout()

    filename = "hmm_3_state_probabilities.png"

    output = FIGURES_DIR / filename

    plt.savefig(output, dpi=300, bbox_inches="tight")

    plt.close()

    print("Saved:", filename)


# ============================================================
# 6. GENERATE ALL FIGURES
# ============================================================

print("\n========== GENERATING FIGURES ==========")

plot_price_regimes(data_2, 2)

plot_price_regimes(data_3, 3)

plot_regime_timeline(data_2, 2)

plot_regime_timeline(data_3, 3)

plot_posterior_probabilities(regime_3)

print("\n========== COMPLETED ==========")

print("All HMM visualisations generated successfully.")

print("\nOutput directory:")
print(FIGURES_DIR)
