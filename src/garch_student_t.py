
"""
Experiment 4B: Student's t-GARCH(1,1)
NIFTY 50 Financial Time-Series Analysis

Objectives:
1. Estimate Gaussian GARCH(1,1).
2. Estimate Student's t-GARCH(1,1).
3. Compare model fit using Log-Likelihood, AIC and BIC.
4. Examine volatility persistence.
5. Compare conditional volatility estimates.
6. Perform residual diagnostic tests.
7. Examine whether Student's t innovations improve
   the modelling of heavy-tailed financial returns.
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


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "processed" / "nifty50.csv"

RESULT_DIR = BASE_DIR / "results" / "garch_student_t"
FIGURE_DIR = BASE_DIR / "figures" / "garch_student_t"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252
ARCH_LAGS = 10

np.random.seed(42)


# ============================================================
# 2. LOAD AND PREPARE DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

if "Log_Return" not in df.columns:
    raise ValueError("Log_Return column not found.")

returns = pd.to_numeric(
    df["Log_Return"], errors="coerce"
).dropna()

if not np.isfinite(returns).all():
    raise ValueError("Returns contain non-finite values.")

if "Date" in df.columns:
    dates = pd.to_datetime(
        df.loc[returns.index, "Date"]
    ).reset_index(drop=True)
else:
    dates = pd.Series(returns.index)

returns = returns.reset_index(drop=True)

# Express returns in percentage points.
returns_pct = returns * 100

print("\n========== DATA INFORMATION ==========")
print("Dataset              : NIFTY 50")
print(f"Observations         : {len(returns_pct)}")
print(f"Mean daily return    : {returns.mean():.8f}")
print(f"Daily return SD      : {returns.std():.8f}")


# ============================================================
# 3. ESTIMATE GAUSSIAN GARCH(1,1)
# ============================================================

print("\nEstimating Gaussian GARCH(1,1)...")

gaussian_model = arch_model(
    returns_pct,
    mean="Constant",
    vol="GARCH",
    p=1,
    q=1,
    dist="normal",
    rescale=False
)

gaussian_result = gaussian_model.fit(
    update_freq=0,
    disp="off"
)


# ============================================================
# 4. ESTIMATE STUDENT'S t-GARCH(1,1)
# ============================================================

print("Estimating Student's t-GARCH(1,1)...")

student_model = arch_model(
    returns_pct,
    mean="Constant",
    vol="GARCH",
    p=1,
    q=1,
    dist="t",
    rescale=False
)

student_result = student_model.fit(
    update_freq=0,
    disp="off"
)

print("\nStudent's t-GARCH estimation completed.")


# ============================================================
# 5. EXTRACT PARAMETERS
# ============================================================

def extract_parameters(result, model_name):

    params = result.params

    mu = params["mu"]
    omega = params["omega"]
    alpha = params["alpha[1]"]
    beta = params["beta[1]"]

    persistence = alpha + beta

    # Long-run variance exists when persistence < 1.
    if persistence < 1:
        unconditional_variance = (
            omega / (1 - persistence)
        )

        unconditional_volatility = np.sqrt(
            unconditional_variance
        )

        unconditional_annual_volatility = (
            unconditional_volatility
            / 100
            * np.sqrt(TRADING_DAYS)
        )

    else:
        unconditional_variance = np.nan
        unconditional_volatility = np.nan
        unconditional_annual_volatility = np.nan

    # Conditional volatility is in percentage points.
    conditional_volatility = (
        result.conditional_volatility.to_numpy()
    )

    conditional_daily_decimal = (
        conditional_volatility / 100
    )

    conditional_annual_decimal = (
        conditional_daily_decimal
        * np.sqrt(TRADING_DAYS)
    )

    # Student's t degrees of freedom.
    nu = (
        params["nu"]
        if "nu" in params.index
        else np.nan
    )

    return {
        "Model": model_name,
        "mu": mu,
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "alpha_plus_beta": persistence,
        "nu": nu,
        "unconditional_variance": unconditional_variance,
        "unconditional_daily_volatility_decimal": (
            unconditional_volatility / 100
            if np.isfinite(unconditional_volatility)
            else np.nan
        ),
        "unconditional_annual_volatility_decimal": (
            unconditional_annual_volatility
        ),
        "mean_conditional_daily_volatility_decimal": (
            conditional_daily_decimal.mean()
        ),
        "mean_conditional_annual_volatility_decimal": (
            conditional_annual_decimal.mean()
        ),
        "log_likelihood": result.loglikelihood,
        "AIC": result.aic,
        "BIC": result.bic,
        "conditional_volatility_pct": conditional_volatility,
        "conditional_annual_volatility": (
            conditional_annual_decimal
        )
    }


gaussian = extract_parameters(
    gaussian_result,
    "Gaussian GARCH(1,1)"
)

student = extract_parameters(
    student_result,
    "Student's t-GARCH(1,1)"
)


# ============================================================
# 6. PRINT MODEL PARAMETERS
# ============================================================

print("\n========== GAUSSIAN GARCH PARAMETERS ==========")

for key in [
    "mu",
    "omega",
    "alpha",
    "beta",
    "alpha_plus_beta",
    "log_likelihood",
    "AIC",
    "BIC"
]:
    print(f"{key:45s}: {gaussian[key]:.8f}")

print("\n========== STUDENT'S t-GARCH PARAMETERS ==========")

for key in [
    "mu",
    "omega",
    "alpha",
    "beta",
    "alpha_plus_beta",
    "nu",
    "log_likelihood",
    "AIC",
    "BIC"
]:
    print(f"{key:45s}: {student[key]:.8f}")

print("\n========== STUDENT'S t VOLATILITY ==========")

print(
    "Mean conditional daily volatility: "
    f"{student['mean_conditional_daily_volatility_decimal']:.6f}"
)

print(
    "Mean conditional annual volatility: "
    f"{student['mean_conditional_annual_volatility_decimal']:.6f}"
)

print(
    "Unconditional annual volatility: "
    f"{student['unconditional_annual_volatility_decimal']:.6f}"
)


# ============================================================
# 7. MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame([
    {
        "Model": gaussian["Model"],
        "Log_Likelihood": gaussian["log_likelihood"],
        "AIC": gaussian["AIC"],
        "BIC": gaussian["BIC"],
        "Alpha": gaussian["alpha"],
        "Beta": gaussian["beta"],
        "Alpha_Plus_Beta": gaussian["alpha_plus_beta"],
        "Degrees_of_Freedom": np.nan
    },
    {
        "Model": student["Model"],
        "Log_Likelihood": student["log_likelihood"],
        "AIC": student["AIC"],
        "BIC": student["BIC"],
        "Alpha": student["alpha"],
        "Beta": student["beta"],
        "Alpha_Plus_Beta": student["alpha_plus_beta"],
        "Degrees_of_Freedom": student["nu"]
    }
])

comparison.to_csv(
    RESULT_DIR / "garch_model_comparison.csv",
    index=False
)

print("\n========== MODEL COMPARISON ==========")
print(comparison.to_string(index=False))

print("\nLower AIC and BIC indicate a better fit,")
print("when comparing models fitted to the same observations.")


# ============================================================
# 8. SAVE PARAMETER ESTIMATES
# ============================================================

parameter_rows = []

for model_info in [gaussian, student]:

    for key in [
        "mu",
        "omega",
        "alpha",
        "beta",
        "alpha_plus_beta",
        "nu",
        "unconditional_daily_volatility_decimal",
        "unconditional_annual_volatility_decimal",
        "mean_conditional_daily_volatility_decimal",
        "mean_conditional_annual_volatility_decimal"
    ]:

        parameter_rows.append({
            "Model": model_info["Model"],
            "Parameter": key,
            "Estimate": model_info[key]
        })

parameter_df = pd.DataFrame(parameter_rows)

parameter_df.to_csv(
    RESULT_DIR / "garch_parameter_estimates.csv",
    index=False
)


# ============================================================
# 9. STANDARDISED RESIDUALS
# ============================================================

gaussian_resid = (
    gaussian_result.std_resid.dropna().to_numpy()
)

student_resid = (
    student_result.std_resid.dropna().to_numpy()
)

gaussian_resid = gaussian_resid[
    np.isfinite(gaussian_resid)
]

student_resid = student_resid[
    np.isfinite(student_resid)
]


# ============================================================
# 10. RESIDUAL DIAGNOSTICS
# ============================================================

def residual_diagnostics(residuals, model_name):

    jb_stat, jb_pvalue = stats.jarque_bera(
        residuals
    )

    lb_resid = acorr_ljungbox(
        residuals,
        lags=[5, 10, 20],
        return_df=True
    )

    lb_squared = acorr_ljungbox(
        residuals ** 2,
        lags=[5, 10, 20],
        return_df=True
    )

    arch_stat, arch_pvalue, _, _ = het_arch(
        residuals,
        nlags=ARCH_LAGS
    )

    return {
        "Model": model_name,
        "Residual_Mean": np.mean(residuals),
        "Residual_SD": np.std(residuals, ddof=1),
        "Skewness": stats.skew(residuals),
        "Excess_Kurtosis": stats.kurtosis(residuals),
        "JB_Statistic": jb_stat,
        "JB_P_Value": jb_pvalue,
        "LB_Residual_Lag5_P": lb_resid.loc[5, "lb_pvalue"],
        "LB_Residual_Lag10_P": lb_resid.loc[10, "lb_pvalue"],
        "LB_Residual_Lag20_P": lb_resid.loc[20, "lb_pvalue"],
        "LB_Squared_Lag5_P": lb_squared.loc[5, "lb_pvalue"],
        "LB_Squared_Lag10_P": lb_squared.loc[10, "lb_pvalue"],
        "LB_Squared_Lag20_P": lb_squared.loc[20, "lb_pvalue"],
        "ARCH_LM_Statistic": arch_stat,
        "ARCH_LM_P_Value": arch_pvalue
    }


gaussian_diagnostics = residual_diagnostics(
    gaussian_resid,
    "Gaussian GARCH(1,1)"
)

student_diagnostics = residual_diagnostics(
    student_resid,
    "Student's t-GARCH(1,1)"
)

diagnostics_df = pd.DataFrame([
    gaussian_diagnostics,
    student_diagnostics
])

diagnostics_df.to_csv(
    RESULT_DIR / "garch_residual_diagnostics.csv",
    index=False
)

print("\n========== RESIDUAL DIAGNOSTIC COMPARISON ==========")

print(
    diagnostics_df.to_string(index=False)
)


# ============================================================
# 11. FIGURE 1: CONDITIONAL VOLATILITY COMPARISON
# ============================================================

fig, ax = plt.subplots(figsize=(15, 6))

ax.plot(
    dates,
    gaussian["conditional_annual_volatility"] * 100,
    linewidth=1.1,
    label="Gaussian GARCH"
)

ax.plot(
    dates,
    student["conditional_annual_volatility"] * 100,
    linewidth=1.1,
    label="Student's t-GARCH"
)

ax.set_title(
    "Gaussian GARCH vs Student's t-GARCH: "
    "Annualised Conditional Volatility"
)

ax.set_xlabel("Date")
ax.set_ylabel("Annualised volatility (%)")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "garch_volatility_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# 12. FIGURE 2: RESIDUAL DISTRIBUTION COMPARISON
# ============================================================

fig, ax = plt.subplots(figsize=(11, 6))

ax.hist(
    gaussian_resid,
    bins=60,
    density=True,
    alpha=0.35,
    label="Gaussian GARCH residuals"
)

ax.hist(
    student_resid,
    bins=60,
    density=True,
    alpha=0.35,
    label="Student's t-GARCH residuals"
)

x = np.linspace(-6, 6, 500)

ax.plot(
    x,
    stats.norm.pdf(x),
    linewidth=2,
    label="Standard normal density"
)

# Standardised Student's t density:
# Y = T / sqrt(nu / (nu - 2))
nu = student["nu"]

if nu > 2:
    scale = np.sqrt((nu - 2) / nu)

    student_density = (
        stats.t.pdf(x / scale, df=nu) / scale
    )

    ax.plot(
        x,
        student_density,
        linewidth=2,
        label=f"Standardised Student's t (df={nu:.2f})"
    )

ax.set_title(
    "Standardised Residual Distribution Comparison"
)

ax.set_xlabel("Standardised residual")
ax.set_ylabel("Probability density")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "garch_residual_distribution_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# 13. FIGURE 3: STUDENT'S t-GARCH Q-Q PLOT
# ============================================================

# Compare sample residual quantiles against theoretical
# quantiles of the fitted variance-standardised Student's t.

n = len(student_resid)

probabilities = (
    np.arange(1, n + 1) - 0.5
) / n

student_theoretical_quantiles = (
    stats.t.ppf(probabilities, df=nu)
    * np.sqrt((nu - 2) / nu)
)

sample_quantiles = np.sort(student_resid)

fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(
    student_theoretical_quantiles,
    sample_quantiles,
    s=12,
    alpha=0.5,
    label="Student's t-GARCH residuals"
)

lower = min(
    student_theoretical_quantiles.min(),
    sample_quantiles.min()
)

upper = max(
    student_theoretical_quantiles.max(),
    sample_quantiles.max()
)

ax.plot(
    [lower, upper],
    [lower, upper],
    linestyle="--",
    linewidth=2,
    label="45-degree reference line"
)

ax.set_title(
    "Student's t Q-Q Plot of Standardised Residuals"
)

ax.set_xlabel("Theoretical standardised t quantiles")
ax.set_ylabel("Sample residual quantiles")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "student_t_garch_qqplot.png",
    dpi=300
)

plt.close()


# ============================================================
# 14. FIGURE 4: CONDITIONAL VOLATILITY DIFFERENCE
# ============================================================

volatility_difference = (
    student["conditional_annual_volatility"]
    - gaussian["conditional_annual_volatility"]
) * 100

fig, ax = plt.subplots(figsize=(15, 5))

ax.plot(
    dates,
    volatility_difference,
    linewidth=1
)

ax.axhline(
    0,
    linestyle="--",
    linewidth=1
)

ax.set_title(
    "Difference in Annualised Conditional Volatility: "
    "Student's t-GARCH minus Gaussian GARCH"
)

ax.set_xlabel("Date")
ax.set_ylabel("Volatility difference (percentage points)")
ax.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "garch_volatility_difference.png",
    dpi=300
)

plt.close()


# ============================================================
# 15. SAVE CONDITIONAL VOLATILITY SERIES
# ============================================================

volatility_df = pd.DataFrame({
    "Date": dates,
    "Actual_Log_Return": returns,
    "Gaussian_Conditional_Volatility_Annual": (
        gaussian["conditional_annual_volatility"]
    ),
    "Student_t_Conditional_Volatility_Annual": (
        student["conditional_annual_volatility"]
    ),
    "Volatility_Difference": (
        student["conditional_annual_volatility"]
        - gaussian["conditional_annual_volatility"]
    )
})

volatility_df.to_csv(
    RESULT_DIR / "garch_conditional_volatility_comparison.csv",
    index=False
)


# ============================================================
# 16. SAVE MODEL SUMMARIES
# ============================================================

with open(
    RESULT_DIR / "garch_student_t_model_summary.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("GAUSSIAN GARCH(1,1)\n")
    file.write("=" * 70 + "\n")
    file.write(str(gaussian_result.summary()))

    file.write("\n\nSTUDENT'S t-GARCH(1,1)\n")
    file.write("=" * 70 + "\n")
    file.write(str(student_result.summary()))

    file.write("\n\nMODEL COMPARISON\n")
    file.write("=" * 70 + "\n")
    file.write(comparison.to_string(index=False))

    file.write("\n\nRESIDUAL DIAGNOSTICS\n")
    file.write("=" * 70 + "\n")
    file.write(diagnostics_df.to_string(index=False))


# ============================================================
# 17. FINAL RESULTS
# ============================================================

print("\n========== OUTPUT FILES ==========")

print("\nFigures:")

for path in sorted(FIGURE_DIR.glob("*.png")):
    print(path.name)

print("\nResults:")

for path in sorted(RESULT_DIR.iterdir()):
    print(path.name)

print("\nStudent's t-GARCH experiment completed successfully.")
