
# ============================================================
# PROJECT: RANDOM WALK VS BROWNIAN MOTION VS REAL STOCK MARKETS
# EXPERIMENT 2: STANDARD BROWNIAN MOTION
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy import stats


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "processed" / "nifty50.csv"

RESULTS_DIR = BASE_DIR / "results" / "brownian_motion"
FIGURES_DIR = BASE_DIR / "figures" / "brownian_motion"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
NUM_PATHS = 1000
TRADING_DAYS = 252

np.random.seed(RANDOM_SEED)


# ============================================================
# 2. LOAD AND PREPARE DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

required_columns = ["Date", "Adj_Close", "Log_Return"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)

returns = df["Log_Return"].dropna().to_numpy(dtype=float)
prices = df["Adj_Close"].to_numpy(dtype=float)

if len(returns) == 0:
    raise ValueError("No valid log returns found.")

if np.any(~np.isfinite(returns)):
    raise ValueError("Log returns contain non-finite values.")

if np.any(~np.isfinite(prices)) or np.any(prices <= 0):
    raise ValueError("Prices must be finite and positive.")


# ============================================================
# 3. ESTIMATE HISTORICAL PARAMETERS
# ============================================================

daily_mu = np.mean(returns)
daily_sigma = np.std(returns, ddof=1)

dt = 1 / TRADING_DAYS
n_steps = len(returns)
total_time = n_steps * dt

print("\n========== BROWNIAN MOTION PARAMETERS ==========")

print(f"Initial NIFTY 50 price : {prices[0]:.4f}")
print(f"Final NIFTY 50 price   : {prices[-1]:.4f}")
print(f"Number of returns      : {n_steps}")
print(f"Daily historical mean  : {daily_mu:.8f}")
print(f"Daily historical SD    : {daily_sigma:.8f}")
print(f"Time increment (dt)    : {dt:.8f}")
print(f"Total simulation time  : {total_time:.4f} years")
print(f"Number of paths        : {NUM_PATHS}")


# ============================================================
# 4. SIMULATE STANDARD BROWNIAN MOTION
# ============================================================

# Generate independent standard normal random variables.
Z = np.random.normal(
    loc=0.0,
    scale=1.0,
    size=(NUM_PATHS, n_steps)
)

# Brownian increments:
# Delta W ~ N(0, dt)

brownian_increments = np.sqrt(dt) * Z

# Construct Brownian paths, beginning at W(0) = 0.
brownian_paths = np.concatenate(
    [
        np.zeros((NUM_PATHS, 1)),
        np.cumsum(brownian_increments, axis=1)
    ],
    axis=1
)

time_grid = np.arange(n_steps + 1) * dt

print("\n========== SIMULATION ==========")
print(f"Number of simulated paths : {NUM_PATHS}")
print(f"Time steps per path       : {n_steps}")
print(f"Brownian path matrix      : {brownian_paths.shape}")


# ============================================================
# 5. TERMINAL VALUES AND THEORETICAL DISTRIBUTION
# ============================================================

terminal_values = brownian_paths[:, -1]

theoretical_terminal_mean = 0.0
theoretical_terminal_sd = np.sqrt(total_time)

theoretical_q05 = stats.norm.ppf(
    0.05,
    loc=0,
    scale=theoretical_terminal_sd
)

theoretical_q95 = stats.norm.ppf(
    0.95,
    loc=0,
    scale=theoretical_terminal_sd
)

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
        np.mean(terminal_values),
        np.median(terminal_values),
        np.std(terminal_values, ddof=1),
        np.percentile(terminal_values, 5),
        np.percentile(terminal_values, 95),
        np.min(terminal_values),
        np.max(terminal_values)
    ],
    "Theoretical": [
        theoretical_terminal_mean,
        0.0,
        theoretical_terminal_sd,
        theoretical_q05,
        theoretical_q95,
        np.nan,
        np.nan
    ]
})

terminal_summary.to_csv(
    RESULTS_DIR / "brownian_terminal_summary.csv",
    index=False
)

print("\n========== TERMINAL VALUE SUMMARY ==========")
print(terminal_summary.round(5).to_string(index=False))


# ============================================================
# 6. FIGURE 1: BROWNIAN MOTION SAMPLE PATHS
# ============================================================

lower_band = np.percentile(brownian_paths, 5, axis=0)
upper_band = np.percentile(brownian_paths, 95, axis=0)
median_path = np.median(brownian_paths, axis=0)

fig, ax = plt.subplots(figsize=(15, 7))

# Display selected sample paths.
for i in range(20):
    ax.plot(
        time_grid,
        brownian_paths[i],
        linewidth=0.8,
        alpha=0.35
    )

# Display pointwise percentile range.
ax.fill_between(
    time_grid,
    lower_band,
    upper_band,
    alpha=0.20,
    label="Simulated 5th-95th percentile range"
)

ax.plot(
    time_grid,
    median_path,
    linewidth=2.2,
    label="Simulated median path"
)

ax.axhline(
    0,
    linestyle="--",
    linewidth=1,
    label="Zero reference"
)

ax.set_title(
    "Simulated Standard Brownian Motion Paths",
    fontsize=15
)

ax.set_xlabel("Time (Years)")
ax.set_ylabel("Brownian Motion Value")

ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "brownian_motion_paths.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 7. FIGURE 2: TERMINAL VALUE DISTRIBUTION
# ============================================================

fig, ax = plt.subplots(figsize=(12, 7))

ax.hist(
    terminal_values,
    bins=35,
    density=True,
    alpha=0.65,
    edgecolor="black",
    label="Simulated terminal values"
)

x = np.linspace(
    min(terminal_values.min(), theoretical_q05 - 2),
    max(terminal_values.max(), theoretical_q95 + 2),
    500
)

theoretical_density = stats.norm.pdf(
    x,
    loc=0,
    scale=theoretical_terminal_sd
)

ax.plot(
    x,
    theoretical_density,
    linewidth=2.5,
    label="Theoretical Normal N(0, T)"
)

ax.axvline(
    np.mean(terminal_values),
    linestyle="--",
    linewidth=1.8,
    label="Simulated mean"
)

ax.axvline(
    0,
    linestyle=":",
    linewidth=1.8,
    label="Theoretical mean"
)

ax.set_title(
    "Distribution of Brownian Motion Terminal Values",
    fontsize=15
)

ax.set_xlabel("Terminal Brownian Motion Value")
ax.set_ylabel("Probability Density")

ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "brownian_terminal_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 8. STANDARDISED INCREMENT COMPARISON
# ============================================================

# Standard Brownian increments have:
# Delta W / sqrt(dt) ~ N(0, 1)

standardised_brownian_increments = (
    brownian_increments / np.sqrt(dt)
).ravel()

# Standardise actual historical returns using historical
# sample mean and sample standard deviation.

standardised_actual_returns = (
    returns - daily_mu
) / daily_sigma


# ============================================================
# 9. FIGURE 3: STANDARDISED INCREMENT DISTRIBUTIONS
# ============================================================

fig, ax = plt.subplots(figsize=(12, 7))

# Use common bin edges for both distributions.
combined_values = np.concatenate([
    standardised_actual_returns,
    standardised_brownian_increments
])

bin_edges = np.histogram_bin_edges(
    combined_values,
    bins=100
)

ax.hist(
    standardised_actual_returns,
    bins=bin_edges,
    density=True,
    alpha=0.60,
    label="Standardised actual NIFTY 50 returns"
)

ax.hist(
    standardised_brownian_increments,
    bins=bin_edges,
    density=True,
    alpha=0.50,
    label="Standardised Brownian increments"
)

x = np.linspace(-5, 5, 500)

ax.plot(
    x,
    stats.norm.pdf(x),
    linestyle="--",
    linewidth=2,
    label="Theoretical Standard Normal"
)



ax.set_title(
    "Standardised Historical Returns vs Brownian Increments",
    fontsize=15
)

ax.set_xlabel("Standardised Daily Increment")
ax.set_ylabel("Probability Density")

ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "brownian_increment_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 10. STATISTICAL COMPARISON
# ============================================================

def calculate_statistics(values):

    jb_result = stats.jarque_bera(values)

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
        "JB Statistic": jb_result.statistic,
        "JB p-value": jb_result.pvalue
    }


actual_stats = calculate_statistics(
    standardised_actual_returns
)

brownian_stats = calculate_statistics(
    standardised_brownian_increments
)

comparison = pd.DataFrame({
    "Statistic": actual_stats.keys(),
    "Actual NIFTY 50": actual_stats.values(),
    "Standard Brownian Motion": brownian_stats.values()
})

comparison.to_csv(
    RESULTS_DIR / "brownian_increment_comparison.csv",
    index=False
)

print("\n========== STANDARDISED INCREMENT COMPARISON ==========")
print(comparison.round(6).to_string(index=False))


# ============================================================
# 11. SAVE SIMULATION PARAMETERS
# ============================================================

parameters = pd.DataFrame({
    "Parameter": [
        "Number of simulated paths",
        "Number of time steps",
        "Trading days per year",
        "Time increment",
        "Total simulation time (years)",
        "Theoretical terminal mean",
        "Theoretical terminal standard deviation",
        "Historical daily mean",
        "Historical daily standard deviation",
        "Random seed"
    ],
    "Value": [
        NUM_PATHS,
        n_steps,
        TRADING_DAYS,
        dt,
        total_time,
        theoretical_terminal_mean,
        theoretical_terminal_sd,
        daily_mu,
        daily_sigma,
        RANDOM_SEED
    ]
})

parameters.to_csv(
    RESULTS_DIR / "brownian_parameters.csv",
    index=False
)


# ============================================================
# 12. FINAL OUTPUT
# ============================================================

print("\n========== OUTPUT FILES ==========")

print("\nFigures saved:")
for file in sorted(FIGURES_DIR.glob("*.png")):
    print(file.name)

print("\nResults saved:")
for file in sorted(RESULTS_DIR.glob("*.csv")):
    print(file.name)

print("\nBrownian Motion simulation completed successfully.")
