
# ============================================================
# PROJECT: RANDOM WALK VS BROWNIAN MOTION VS REAL STOCK MARKETS
# EXPERIMENT 3: GEOMETRIC BROWNIAN MOTION (GBM)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy import stats


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "processed" / "nifty50.csv"

RESULTS_DIR = BASE_DIR / "results" / "gbm"
FIGURES_DIR = BASE_DIR / "figures" / "gbm"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
NUM_PATHS = 1000
TRADING_DAYS = 252

np.random.seed(RANDOM_SEED)


# ============================================================
# 2. LOAD HISTORICAL DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

prices = df["Adj_Close"].to_numpy(dtype=float)
returns = df["Log_Return"].dropna().to_numpy(dtype=float)

if len(returns) == 0:
    raise ValueError("No valid historical returns found.")

if np.any(~np.isfinite(returns)):
    raise ValueError("Returns contain non-finite values.")

if np.any(~np.isfinite(prices)) or np.any(prices <= 0):
    raise ValueError("Prices must be finite and positive.")

initial_price = prices[0]
actual_final_price = prices[-1]

n_steps = len(returns)


# ============================================================
# 3. ESTIMATE GBM PARAMETERS
# ============================================================

# Daily historical log-return mean and sample standard deviation.
daily_mean = np.mean(returns)
daily_sd = np.std(returns, ddof=1)

dt = 1 / TRADING_DAYS
total_time = n_steps * dt

# Under GBM:
# log(S_t/S_0) ~ N((mu - sigma^2/2)t, sigma^2 t)
#
# Therefore:
# mu = 252 * daily_mean + sigma_annual^2 / 2

annual_sigma = daily_sd * np.sqrt(TRADING_DAYS)

annual_mu = (
    TRADING_DAYS * daily_mean
    + 0.5 * annual_sigma**2
)

annual_log_drift = annual_mu - 0.5 * annual_sigma**2

print("\n========== GBM PARAMETERS ==========")

print(f"Initial NIFTY 50 price : {initial_price:.4f}")
print(f"Actual final price     : {actual_final_price:.4f}")
print(f"Number of returns      : {n_steps}")
print(f"Daily mean log return  : {daily_mean:.8f}")
print(f"Daily return SD        : {daily_sd:.8f}")
print(f"Annual drift (mu)      : {annual_mu:.6f}")
print(f"Annual volatility      : {annual_sigma:.6f}")
print(f"Annual log drift       : {annual_log_drift:.6f}")
print(f"Time increment (dt)    : {dt:.8f}")
print(f"Total time (years)     : {total_time:.4f}")


# ============================================================
# 4. SIMULATE GEOMETRIC BROWNIAN MOTION
# ============================================================

# Exact GBM discretisation:
#
# S(t+dt) = S(t) * exp(
#       (mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z
# )
#
# Z ~ N(0,1)

Z = np.random.normal(
    loc=0,
    scale=1,
    size=(NUM_PATHS, n_steps)
)

log_price_increments = (
    annual_log_drift * dt
    + annual_sigma * np.sqrt(dt) * Z
)

log_price_paths = np.cumsum(
    log_price_increments,
    axis=1
)

simulated_prices = initial_price * np.exp(
    np.column_stack([
        np.zeros(NUM_PATHS),
        log_price_paths
    ])
)

time_grid = np.arange(n_steps + 1) / TRADING_DAYS

print("\n========== SIMULATION ==========")
print(f"Number of paths       : {NUM_PATHS}")
print(f"Trading days per path : {n_steps}")
print(f"Price matrix shape    : {simulated_prices.shape}")
print(f"Minimum simulated price: {simulated_prices.min():.4f}")

if np.any(simulated_prices <= 0):
    raise ValueError("GBM generated a non-positive price.")


# ============================================================
# 5. THEORETICAL TERMINAL PRICE DISTRIBUTION
# ============================================================

# Under GBM:
#
# ln(S_T/S_0) ~ N(
#     (mu - sigma^2/2)*T,
#     sigma^2*T
# )

theoretical_log_mean = (
    annual_log_drift * total_time
)

theoretical_log_sd = (
    annual_sigma * np.sqrt(total_time)
)

theoretical_terminal_mean = (
    initial_price
    * np.exp(annual_mu * total_time)
)

theoretical_terminal_median = (
    initial_price
    * np.exp(theoretical_log_mean)
)

theoretical_terminal_sd = (
    theoretical_terminal_mean
    * np.sqrt(
        np.exp(annual_sigma**2 * total_time) - 1
    )
)

terminal_prices = simulated_prices[:, -1]

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
    "Simulated": [
        np.mean(terminal_prices),
        np.median(terminal_prices),
        np.std(terminal_prices, ddof=1),
        np.percentile(terminal_prices, 5),
        np.percentile(terminal_prices, 95),
        np.min(terminal_prices),
        np.max(terminal_prices)
    ],
    "Theoretical": [
        theoretical_terminal_mean,
        theoretical_terminal_median,
        theoretical_terminal_sd,
        initial_price * np.exp(
            theoretical_log_mean
            + stats.norm.ppf(0.05) * theoretical_log_sd
        ),
        initial_price * np.exp(
            theoretical_log_mean
            + stats.norm.ppf(0.95) * theoretical_log_sd
        ),
        np.nan,
        np.nan
    ]
})

terminal_summary.to_csv(
    RESULTS_DIR / "gbm_terminal_summary.csv",
    index=False
)

print("\n========== TERMINAL PRICE SUMMARY ==========")
print(terminal_summary.round(4).to_string(index=False))


# ============================================================
# 6. FIGURE 1: ACTUAL VS SIMULATED PRICE PATHS
# ============================================================

lower_band = np.percentile(
    simulated_prices,
    5,
    axis=0
)

upper_band = np.percentile(
    simulated_prices,
    95,
    axis=0
)

median_path = np.median(
    simulated_prices,
    axis=0
)

fig, ax = plt.subplots(figsize=(15, 7))

ax.plot(
    time_grid,
    prices,
    linewidth=2.2,
    label="Actual NIFTY 50"
)

ax.plot(
    time_grid,
    median_path,
    linewidth=2,
    label="GBM median path"
)

ax.fill_between(
    time_grid,
    lower_band,
    upper_band,
    alpha=0.20,
    label="GBM 5th-95th percentile range"
)

ax.set_yscale("log")

ax.set_title(
    "Actual NIFTY 50 vs Geometric Brownian Motion",
    fontsize=15
)

ax.set_xlabel("Time (Years)")
ax.set_ylabel("Index Level (Log Scale)")

ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "gbm_price_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 7. FIGURE 2: TERMINAL PRICE DISTRIBUTION
# ============================================================

fig, ax = plt.subplots(figsize=(12, 7))

log_terminal_prices = np.log10(terminal_prices)

ax.hist(
    log_terminal_prices,
    bins=40,
    density=True,
    alpha=0.65,
    edgecolor="black",
    label="Simulated log10 terminal prices"
)

x_log = np.linspace(
    log_terminal_prices.min(),
    log_terminal_prices.max(),
    500
)

x_prices = 10**x_log

# Transform the theoretical lognormal density
# into the density of log10(S_T).
#
# If Y = log10(S_T), then:
# f_Y(y) = ln(10)*S_T*f_S(S_T)

theoretical_density_log = (
    np.log(10)
    * x_prices
    * stats.lognorm.pdf(
        x_prices,
        s=theoretical_log_sd,
        scale=initial_price * np.exp(theoretical_log_mean)
    )
)

ax.plot(
    x_log,
    theoretical_density_log,
    linewidth=2.5,
    label="Theoretical lognormal distribution"
)

ax.axvline(
    np.log10(actual_final_price),
    linestyle="--",
    linewidth=2,
    label="Actual final NIFTY 50 price"
)

ax.set_title(
    "Distribution of GBM Terminal Prices",
    fontsize=15
)

ax.set_xlabel("Log10(Terminal Price)")
ax.set_ylabel("Probability Density")

ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "gbm_terminal_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 8. FIGURE 3: HISTORICAL VS SIMULATED RETURNS
# ============================================================

# Recover simulated daily log returns from price paths.
simulated_log_returns = np.diff(
    np.log(simulated_prices),
    axis=1
).ravel()

combined_returns = np.concatenate([
    returns,
    simulated_log_returns
])

bin_edges = np.histogram_bin_edges(
    combined_returns,
    bins=100
)

fig, ax = plt.subplots(figsize=(12, 7))

ax.hist(
    returns,
    bins=bin_edges,
    density=True,
    alpha=0.60,
    label="Actual NIFTY 50 log returns"
)

ax.hist(
    simulated_log_returns,
    bins=bin_edges,
    density=True,
    alpha=0.50,
    label="Simulated GBM log returns"
)

ax.set_title(
    "Actual vs Simulated GBM Daily Log Returns",
    fontsize=15
)

ax.set_xlabel("Daily Log Return")
ax.set_ylabel("Probability Density")

ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "gbm_return_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 9. STATISTICAL COMPARISON
# ============================================================

def calculate_statistics(values):

    jb = stats.jarque_bera(values)

    return {
        "Mean": np.mean(values),
        "Standard Deviation": np.std(values, ddof=1),
        "Skewness": stats.skew(values, bias=False),
        "Excess Kurtosis": stats.kurtosis(
            values,
            fisher=True,
            bias=False
        ),
        "Minimum": np.min(values),
        "Maximum": np.max(values),
        "5th Percentile": np.percentile(values, 5),
        "50th Percentile": np.percentile(values, 50),
        "95th Percentile": np.percentile(values, 95),
        "JB Statistic": jb.statistic,
        "JB p-value": jb.pvalue
    }


actual_stats = calculate_statistics(returns)

simulated_stats = calculate_statistics(
    simulated_log_returns
)

comparison = pd.DataFrame({
    "Statistic": actual_stats.keys(),
    "Actual NIFTY 50": actual_stats.values(),
    "Simulated GBM": simulated_stats.values()
})

comparison.to_csv(
    RESULTS_DIR / "gbm_return_comparison.csv",
    index=False
)

print("\n========== RETURN STATISTICAL COMPARISON ==========")
print(comparison.round(6).to_string(index=False))


# ============================================================
# 10. SAVE MODEL PARAMETERS
# ============================================================

parameters = pd.DataFrame({
    "Parameter": [
        "Number of paths",
        "Number of trading days",
        "Trading days per year",
        "Initial price",
        "Daily mean log return",
        "Daily standard deviation",
        "Annual drift",
        "Annual volatility",
        "Annual log drift",
        "Time increment",
        "Total simulation time",
        "Theoretical terminal mean",
        "Theoretical terminal median",
        "Theoretical terminal standard deviation",
        "Random seed"
    ],
    "Value": [
        NUM_PATHS,
        n_steps,
        TRADING_DAYS,
        initial_price,
        daily_mean,
        daily_sd,
        annual_mu,
        annual_sigma,
        annual_log_drift,
        dt,
        total_time,
        theoretical_terminal_mean,
        theoretical_terminal_median,
        theoretical_terminal_sd,
        RANDOM_SEED
    ]
})

parameters.to_csv(
    RESULTS_DIR / "gbm_parameters.csv",
    index=False
)


# ============================================================
# 11. FINAL OUTPUT
# ============================================================

print("\n========== OUTPUT FILES ==========")

print("\nFigures:")
for file in sorted(FIGURES_DIR.glob("*.png")):
    print(file.name)

print("\nResults:")
for file in sorted(RESULTS_DIR.glob("*.csv")):
    print(file.name)

print("\nGeometric Brownian Motion simulation completed successfully.")
