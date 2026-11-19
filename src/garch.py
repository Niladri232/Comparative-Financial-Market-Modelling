
"""
Experiment 4: GARCH(1,1) Modelling of NIFTY 50

Objectives:
1. Estimate GARCH(1,1) parameters.
2. Estimate time-varying conditional volatility.
3. Examine volatility clustering and persistence.
4. Analyse standardised residuals.
5. Perform residual diagnostic tests.
6. Save results and figures.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arch import arch_model
from scipy import stats
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_arch
)
from statsmodels.graphics.gofplots import qqplot


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "processed" / "nifty50.csv"

RESULT_DIR = BASE_DIR / "results" / "garch"
FIGURE_DIR = BASE_DIR / "figures" / "garch"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252
ARCH_LAGS = 10


# ============================================================
# 2. LOAD AND PREPARE DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

if "Log_Return" not in df.columns:
    raise ValueError("Log_Return column not found in dataset.")

returns = pd.to_numeric(
    df["Log_Return"], errors="coerce"
).dropna()

if "Date" in df.columns:
    dates = pd.to_datetime(
        df.loc[returns.index, "Date"]
    )
else:
    dates = returns.index

if len(returns) == 0:
    raise ValueError("No valid return observations found.")

if not np.isfinite(returns).all():
    raise ValueError("Returns contain non-finite values.")

# Convert decimal log returns into percentage points.
returns_pct = returns * 100

print("\n========== DATA INFORMATION ==========")
print(f"Dataset              : NIFTY 50")
print(f"Number of observations: {len(returns)}")
print(f"First daily return    : {returns.iloc[0]:.6f}")
print(f"Last daily return     : {returns.iloc[-1]:.6f}")
print(f"Mean daily return     : {returns.mean():.8f}")
print(f"Daily return SD       : {returns.std():.8f}")


# ============================================================
# 3. ESTIMATE GARCH(1,1)
# ============================================================

# Mean equation:
# r_t = mu + epsilon_t
#
# Variance equation:
# sigma_t^2 = omega
#             + alpha * epsilon_(t-1)^2
#             + beta * sigma_(t-1)^2

model = arch_model(
    returns_pct,
    mean="Constant",
    vol="GARCH",
    p=1,
    q=1,
    dist="normal",
    rescale=False
)

result = model.fit(
    update_freq=0,
    disp="off"
)

print("\n========== GARCH(1,1) MODEL SUMMARY ==========")
print(result.summary())


# ============================================================
# 4. EXTRACT MODEL PARAMETERS
# ============================================================

params = result.params

mu = params["mu"]
omega = params["omega"]
alpha = params["alpha[1]"]
beta = params["beta[1]"]

persistence = alpha + beta

# Returns were multiplied by 100 before estimation.
# Convert variance parameters back to decimal-return units.
omega_decimal = omega / (100 ** 2)

if persistence < 1:
    unconditional_variance = (
        omega / (1 - persistence)
    )
    unconditional_volatility = np.sqrt(
        unconditional_variance
    )
    unconditional_volatility_annual = (
        unconditional_volatility * np.sqrt(TRADING_DAYS)
    )
else:
    unconditional_variance = np.nan
    unconditional_volatility = np.nan
    unconditional_volatility_annual = np.nan

# Conditional volatility from ARCH is in percentage points.
conditional_volatility_pct = (
    result.conditional_volatility
)

# Convert to decimal and annualise.
conditional_volatility_daily = (
    conditional_volatility_pct / 100
)

conditional_volatility_annual = (
    conditional_volatility_daily
    * np.sqrt(TRADING_DAYS)
)

print("\n========== GARCH PARAMETERS ==========")
print(f"Mean parameter (mu)       : {mu:.8f}")
print(f"Omega                     : {omega:.8f}")
print(f"Alpha                     : {alpha:.8f}")
print(f"Beta                      : {beta:.8f}")
print(f"Alpha + Beta              : {persistence:.8f}")

if persistence < 1:
    print(
        "Unconditional daily volatility: "
        f"{unconditional_volatility / 100:.6f}"
    )
    print(
        "Unconditional annual volatility: "
        f"{unconditional_volatility_annual / 100:.6f}"
    )
else:
    print(
        "WARNING: alpha + beta >= 1. "
        "Finite unconditional variance is not available."
    )

print(
    f"Mean conditional daily volatility: "
    f"{conditional_volatility_daily.mean():.6f}"
)

print(
    f"Mean conditional annual volatility: "
    f"{conditional_volatility_annual.mean():.6f}"
)


# ============================================================
# 5. SAVE PARAMETER ESTIMATES
# ============================================================

parameter_table = pd.DataFrame({
    "Parameter": [
        "mu",
        "omega",
        "alpha",
        "beta",
        "alpha_plus_beta",
        "omega_decimal",
        "unconditional_variance_pct_squared",
        "unconditional_daily_volatility_decimal",
        "unconditional_annual_volatility_decimal",
        "log_likelihood",
        "AIC",
        "BIC"
    ],
    "Estimate": [
        mu,
        omega,
        alpha,
        beta,
        persistence,
        omega_decimal,
        unconditional_variance,
        (
            unconditional_volatility / 100
            if np.isfinite(unconditional_volatility)
            else np.nan
        ),
        (
            unconditional_volatility_annual / 100
            if np.isfinite(unconditional_volatility_annual)
            else np.nan
        ),
        result.loglikelihood,
        result.aic,
        result.bic
    ]
})

parameter_table.to_csv(
    RESULT_DIR / "garch_parameters.csv",
    index=False
)


# ============================================================
# 6. CONDITIONAL VOLATILITY DATA
# ============================================================

volatility_df = pd.DataFrame({
    "Date": dates.values,
    "Log_Return": returns.values,
    "Log_Return_Percent": returns_pct.values,
    "Conditional_Volatility_Daily": (
        conditional_volatility_daily.values
    ),
    "Conditional_Volatility_Annual": (
        conditional_volatility_annual.values
    )
})

volatility_df.to_csv(
    RESULT_DIR / "garch_conditional_volatility.csv",
    index=False
)


# ============================================================
# 7. STANDARDISED RESIDUALS
# ============================================================

standardized_residuals = (
    result.std_resid.dropna()
)

standardized_residuals = (
    standardized_residuals[
        np.isfinite(standardized_residuals)
    ]
)

residual_mean = standardized_residuals.mean()
residual_sd = standardized_residuals.std()
residual_skewness = stats.skew(
    standardized_residuals
)
residual_excess_kurtosis = stats.kurtosis(
    standardized_residuals
)

jb_stat, jb_pvalue = stats.jarque_bera(
    standardized_residuals
)

print("\n========== STANDARDISED RESIDUAL STATISTICS ==========")
print(f"Mean              : {residual_mean:.6f}")
print(f"Standard deviation: {residual_sd:.6f}")
print(f"Skewness          : {residual_skewness:.6f}")
print(f"Excess kurtosis   : {residual_excess_kurtosis:.6f}")
print(f"Jarque-Bera       : {jb_stat:.6f}")
print(f"JB p-value        : {jb_pvalue:.6g}")


# ============================================================
# 8. RESIDUAL DIAGNOSTIC TESTS
# ============================================================

# Ljung-Box test on standardised residuals.
lb_residual = acorr_ljungbox(
    standardized_residuals,
    lags=[5, 10, 20],
    return_df=True
)

# Ljung-Box test on squared standardised residuals.
lb_squared = acorr_ljungbox(
    standardized_residuals ** 2,
    lags=[5, 10, 20],
    return_df=True
)

# ARCH-LM test on standardised residuals.
arch_stat, arch_pvalue, _, _ = het_arch(
    standardized_residuals,
    nlags=ARCH_LAGS
)

print("\n========== LJUNG-BOX: STANDARDISED RESIDUALS ==========")
print(lb_residual)

print("\n========== LJUNG-BOX: SQUARED RESIDUALS ==========")
print(lb_squared)

print("\n========== ARCH-LM TEST ==========")
print(f"LM Statistic: {arch_stat:.6f}")
print(f"p-value     : {arch_pvalue:.6g}")


# ============================================================
# 9. SAVE DIAGNOSTIC RESULTS
# ============================================================

diagnostic_rows = []

for lag, row in lb_residual.iterrows():
    diagnostic_rows.append({
        "Test": "Ljung-Box Standardised Residuals",
        "Lag": lag,
        "Statistic": row["lb_stat"],
        "P_Value": row["lb_pvalue"]
    })

for lag, row in lb_squared.iterrows():
    diagnostic_rows.append({
        "Test": "Ljung-Box Squared Residuals",
        "Lag": lag,
        "Statistic": row["lb_stat"],
        "P_Value": row["lb_pvalue"]
    })

diagnostic_rows.append({
    "Test": "ARCH-LM Standardised Residuals",
    "Lag": ARCH_LAGS,
    "Statistic": arch_stat,
    "P_Value": arch_pvalue
})

diagnostic_rows.append({
    "Test": "Jarque-Bera Standardised Residuals",
    "Lag": np.nan,
    "Statistic": jb_stat,
    "P_Value": jb_pvalue
})

diagnostics_df = pd.DataFrame(diagnostic_rows)

diagnostics_df.to_csv(
    RESULT_DIR / "garch_diagnostics.csv",
    index=False
)


# ============================================================
# 10. FIGURE 1: RETURNS AND CONDITIONAL VOLATILITY
# ============================================================

fig, ax = plt.subplots(figsize=(15, 6))

ax.plot(
    dates,
    returns_pct,
    linewidth=0.6,
    alpha=0.55,
    label="NIFTY 50 daily log returns (%)"
)

ax.plot(
    dates,
    conditional_volatility_pct,
    linewidth=1.2,
    label="GARCH conditional volatility (%)"
)

ax.set_title(
    "NIFTY 50 Returns and GARCH Conditional Volatility"
)
ax.set_xlabel("Date")
ax.set_ylabel("Daily percentage points")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "garch_returns_volatility.png",
    dpi=300
)

plt.close()


# ============================================================
# 11. FIGURE 2: ANNUALISED CONDITIONAL VOLATILITY
# ============================================================

fig, ax = plt.subplots(figsize=(15, 6))

ax.plot(
    dates,
    conditional_volatility_annual * 100,
    linewidth=1.2,
    label="GARCH annualised conditional volatility"
)

if np.isfinite(unconditional_volatility_annual):
    ax.axhline(
        unconditional_volatility_annual * 100,
        linestyle="--",
        linewidth=1.5,
        label="Unconditional annualised volatility"
    )

ax.set_title(
    "GARCH(1,1) Annualised Conditional Volatility — NIFTY 50"
)
ax.set_xlabel("Date")
ax.set_ylabel("Annualised volatility (%)")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "garch_conditional_volatility.png",
    dpi=300
)

plt.close()


# ============================================================
# 12. FIGURE 3: STANDARDISED RESIDUAL DISTRIBUTION
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(
    standardized_residuals,
    bins=60,
    density=True,
    alpha=0.65,
    label="GARCH standardised residuals"
)

x = np.linspace(
    standardized_residuals.min(),
    standardized_residuals.max(),
    500
)

ax.plot(
    x,
    stats.norm.pdf(x),
    linewidth=2,
    label="Standard normal density"
)

ax.set_title(
    "Distribution of GARCH Standardised Residuals"
)
ax.set_xlabel("Standardised residual")
ax.set_ylabel("Probability density")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "garch_residual_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 13. FIGURE 4: RESIDUAL Q-Q PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(8, 8))

qqplot(
    standardized_residuals,
    line="45",
    ax=ax
)

ax.set_title(
    "Normal Q-Q Plot of GARCH Standardised Residuals"
)
ax.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "garch_residual_qqplot.png",
    dpi=300
)

plt.close()


# ============================================================
# 14. SAVE TEXT SUMMARY
# ============================================================

summary_file = RESULT_DIR / "garch_model_summary.txt"

with open(summary_file, "w", encoding="utf-8") as file:
    file.write(str(result.summary()))

    file.write("\n\nGARCH(1,1) DIAGNOSTIC RESULTS\n")
    file.write("=" * 55 + "\n")

    file.write(
        f"\nAlpha + Beta: {persistence:.8f}\n"
    )

    file.write(
        f"\nStandardised residual JB p-value: "
        f"{jb_pvalue:.8g}\n"
    )

    file.write(
        f"\nARCH-LM p-value: {arch_pvalue:.8g}\n"
    )

    file.write(
        "\nLjung-Box on standardised residuals:\n"
    )
    file.write(lb_residual.to_string())

    file.write(
        "\n\nLjung-Box on squared standardised residuals:\n"
    )
    file.write(lb_squared.to_string())


# ============================================================
# 15. FINAL OUTPUT
# ============================================================

print("\n========== OUTPUT FILES ==========")

print("\nFigures:")
for path in sorted(FIGURE_DIR.glob("*.png")):
    print(path.name)

print("\nResults:")
for path in sorted(RESULT_DIR.iterdir()):
    print(path.name)

print("\nGARCH(1,1) estimation completed successfully.")
