
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "nifty50.csv"

RESULTS_DIR = PROJECT_ROOT / "results" / "random_walk"
FIGURES_DIR = PROJECT_ROOT / "figures" / "random_walk"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

N_SIMULATIONS = 1000
RANDOM_SEED = 42
TRADING_DAYS = 252


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

required_columns = ["Date", "Adj_Close", "Log_Return"]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Adj_Close"] = pd.to_numeric(
    df["Adj_Close"], errors="coerce"
)
df["Log_Return"] = pd.to_numeric(
    df["Log_Return"], errors="coerce"
)

df = df.sort_values("Date").reset_index(drop=True)

if df["Date"].isna().any():
    raise ValueError("Invalid dates detected.")

if df["Adj_Close"].isna().any() or (df["Adj_Close"] <= 0).any():
    raise ValueError("Invalid adjusted closing prices detected.")

if df["Date"].duplicated().any():
    raise ValueError("Duplicate dates detected.")

prices = df["Adj_Close"].to_numpy(dtype=float)
returns = df["Log_Return"].dropna().to_numpy(dtype=float)
dates = df["Date"]

if not np.isfinite(returns).all():
    raise ValueError("Non-finite log returns detected.")

if len(returns) != len(prices) - 1:
    raise ValueError(
        "The number of returns does not match the price series."
    )


# ============================================================
# 3. PARAMETER ESTIMATION
# ============================================================

mu_hat = np.mean(returns)
sigma_hat = np.std(returns, ddof=1)

initial_price = prices[0]
n_steps = len(returns)

annualized_return = mu_hat * TRADING_DAYS
annualized_volatility = sigma_hat * np.sqrt(TRADING_DAYS)

print("\n========== RANDOM WALK PARAMETERS ==========")

print(f"Initial NIFTY 50 price : {initial_price:.4f}")
print(f"Number of returns      : {n_steps}")
print(f"Daily mean (mu)        : {mu_hat:.8f}")
print(f"Daily volatility       : {sigma_hat:.8f}")
print(f"Annualized return      : {annualized_return:.6f}")
print(f"Annualized volatility  : {annualized_volatility:.6f}")


# ============================================================
# 4. RANDOM WALK SIMULATION
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)

# Generate independent Gaussian innovations.
innovations = rng.normal(
    loc=0.0,
    scale=sigma_hat,
    size=(N_SIMULATIONS, n_steps)
)

# Simulated daily log returns.
simulated_returns = mu_hat + innovations

# Cumulative simulated log returns.
cumulative_returns = np.cumsum(
    simulated_returns,
    axis=1
)

# Simulated price paths.
simulated_prices = initial_price * np.exp(
    cumulative_returns
)

# Add the common initial price to every path.
simulated_prices = np.column_stack([
    np.full(N_SIMULATIONS, initial_price),
    simulated_prices
])

print("\n========== SIMULATION ==========")
print(f"Number of simulated paths : {N_SIMULATIONS}")
print(f"Trading days per path     : {n_steps}")
print(f"Simulated price matrix    : {simulated_prices.shape}")


# ============================================================
# 5. FIGURE 1: ACTUAL PRICE VS SIMULATED PATHS
# ============================================================

time_axis = np.arange(n_steps + 1)

lower_bound = np.percentile(
    simulated_prices, 5, axis=0
)

upper_bound = np.percentile(
    simulated_prices, 95, axis=0
)

median_path = np.median(
    simulated_prices, axis=0
)

fig, ax = plt.subplots(figsize=(13, 6))

# Display a sample of 20 simulated paths.
for i in range(20):
    ax.plot(
        time_axis,
        simulated_prices[i],
        linewidth=0.7,
        alpha=0.25
    )

ax.fill_between(
    time_axis,
    lower_bound,
    upper_bound,
    alpha=0.18,
    label="Simulated 5th–95th percentile range"
)

ax.plot(
    time_axis,
    median_path,
    linewidth=2,
    label="Simulated median path"
)

ax.plot(
    time_axis,
    prices,
    linewidth=1.8,
    label="Actual NIFTY 50"
)

ax.set_title(
    "Actual NIFTY 50 vs Gaussian Random Walk Simulations (Log Scale)"
)

ax.set_xlabel("Trading Days")
ax.set_ylabel("Adjusted Closing Price")
ax.set_yscale("log")

ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()

fig.savefig(
    FIGURES_DIR / "random_walk_price_comparison.png",
    dpi=300
)

plt.close(fig)


# ============================================================
# 6. FIGURE 2: RETURN DISTRIBUTION COMPARISON
# ============================================================

actual_mean = np.mean(returns)
actual_std = np.std(returns, ddof=1)

# Pool all simulated returns for a descriptive comparison.
simulated_returns_flat = simulated_returns.ravel()

fig, ax = plt.subplots(figsize=(11, 6))

# Use common histogram bins for a fair comparison.
combined_returns = np.concatenate([
    returns,
    simulated_returns_flat
])

common_bins = np.linspace(
    combined_returns.min(),
    combined_returns.max(),
    101
)

ax.hist(
    returns,
    bins=common_bins,
    density=True,
    alpha=0.55,
    label="Actual NIFTY 50 returns"
)

ax.hist(
    simulated_returns_flat,
    bins=common_bins,
    density=True,
    alpha=0.45,
    label="Simulated Random Walk returns"
)

ax.set_title("Actual vs Simulated Daily Log Return Distributions")
ax.set_xlabel("Daily Log Return")
ax.set_ylabel("Density")

ax.legend()
ax.grid(alpha=0.2)

plt.tight_layout()

fig.savefig(
    FIGURES_DIR / "random_walk_return_distribution.png",
    dpi=300
)

plt.close(fig)



# ============================================================
# 7. FIGURE 3: TERMINAL PRICE DISTRIBUTION
# ============================================================

terminal_prices = simulated_prices[:, -1]

# Transform terminal prices using base-10 logarithms.
log_terminal_prices = np.log10(terminal_prices)

actual_final_log = np.log10(prices[-1])

median_terminal_log = np.log10(
    np.median(terminal_prices)
)

fig, ax = plt.subplots(figsize=(11, 6))

ax.hist(
    log_terminal_prices,
    bins=35,
    edgecolor="black",
    alpha=0.75
)

ax.axvline(
    actual_final_log,
    linestyle="--",
    linewidth=2,
    label="Actual final NIFTY 50 price"
)

ax.axvline(
    median_terminal_log,
    linestyle=":",
    linewidth=2,
    label="Simulated median terminal price"
)

ax.set_title(
    "Distribution of Simulated Log10 Terminal Prices"
)

ax.set_xlabel("Log10(Terminal Price)")
ax.set_ylabel("Frequency")

ax.legend()
ax.grid(alpha=0.2)

plt.tight_layout()

fig.savefig(
    FIGURES_DIR / "random_walk_terminal_distribution.png",
    dpi=300
)

plt.close(fig)





# ============================================================
# 8. STATISTICAL COMPARISON
# ============================================================

def calculate_statistics(x):

    x = np.asarray(x)

    return {
        "Mean": np.mean(x),
        "Standard Deviation": np.std(x, ddof=1),
        "Skewness": skew(x, bias=False),
        "Excess Kurtosis": kurtosis(
            x, fisher=True, bias=False
        ),
        "Minimum": np.min(x),
        "Maximum": np.max(x),
        "5th Percentile": np.percentile(x, 5),
        "50th Percentile": np.percentile(x, 50),
        "95th Percentile": np.percentile(x, 95)
    }


actual_stats = calculate_statistics(returns)

simulated_stats = calculate_statistics(
    simulated_returns_flat
)

comparison = pd.DataFrame({
    "Actual NIFTY 50": actual_stats,
    "Simulated Random Walk": simulated_stats
})

comparison["Difference (Simulated - Actual)"] = (
    comparison["Simulated Random Walk"]
    - comparison["Actual NIFTY 50"]
)

comparison.to_csv(
    RESULTS_DIR / "random_walk_statistical_comparison.csv"
)

print("\n========== RETURN STATISTICS ==========")
print(comparison.round(6).to_string())


# ============================================================
# 9. TERMINAL PRICE SUMMARY
# ============================================================

terminal_summary = pd.DataFrame({
    "Statistic": [
        "Mean",
        "Median",
        "Standard Deviation",
        "5th Percentile",
        "95th Percentile",
        "Minimum",
        "Maximum"
    ],
    "Simulated Terminal Price": [
        np.mean(terminal_prices),
        np.median(terminal_prices),
        np.std(terminal_prices, ddof=1),
        np.percentile(terminal_prices, 5),
        np.percentile(terminal_prices, 95),
        np.min(terminal_prices),
        np.max(terminal_prices)
    ]
})

terminal_summary.to_csv(
    RESULTS_DIR / "random_walk_terminal_summary.csv",
    index=False
)

print("\n========== TERMINAL PRICE SUMMARY ==========")
print(terminal_summary.round(4).to_string(index=False))


# ============================================================
# 10. SAVE SIMULATION PARAMETERS
# ============================================================

parameters = pd.DataFrame({
    "Parameter": [
        "Initial Price",
        "Daily Mean",
        "Daily Volatility",
        "Annualized Mean",
        "Annualized Volatility",
        "Number of Simulations",
        "Number of Trading Days",
        "Random Seed"
    ],
    "Value": [
        initial_price,
        mu_hat,
        sigma_hat,
        annualized_return,
        annualized_volatility,
        N_SIMULATIONS,
        n_steps,
        RANDOM_SEED
    ]
})

parameters.to_csv(
    RESULTS_DIR / "random_walk_parameters.csv",
    index=False
)

print("\nRandom Walk simulation completed successfully.")

print(f"\nFigures saved in: {FIGURES_DIR}")
print(f"Results saved in: {RESULTS_DIR}")
