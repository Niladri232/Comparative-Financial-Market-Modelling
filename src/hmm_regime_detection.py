
"""
Experiment 5: Hidden Markov Model for Financial Market Regime Detection

Dataset: NIFTY 50 daily log returns
Models: Gaussian HMM with 2 and 3 hidden states

Outputs:
    results/hmm/
    figures/hmm/
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from hmmlearn.hmm import GaussianHMM


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "processed" / "nifty50.csv"

RESULTS_DIR = BASE_DIR / "results" / "hmm"
FIGURES_DIR = BASE_DIR / "figures" / "hmm"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252

N_INITIALISATIONS = 10
MAX_ITERATIONS = 500
CONVERGENCE_TOLERANCE = 1e-4

RANDOM_SEED = 42

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# 2. LOAD AND PREPARE DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df["Log_Return"] = pd.to_numeric(
    df["Log_Return"], errors="coerce"
)

df = df.dropna(subset=["Date", "Log_Return"]).copy()

df = df.sort_values("Date").reset_index(drop=True)

if not np.isfinite(df["Log_Return"]).all():
    raise ValueError("Non-finite values found in Log_Return.")

dates = df["Date"].to_numpy()

# Convert decimal returns into percentage points.
# Example: 0.01 becomes 1.0.
returns = df["Log_Return"].to_numpy(dtype=float)

X = (returns * 100).reshape(-1, 1)

n_observations = len(X)

if n_observations < 100:
    raise ValueError("Insufficient observations for HMM estimation.")

print("\n========== DATA INFORMATION ==========")
print("Dataset                  : NIFTY 50")
print(f"Number of observations   : {n_observations}")
print(f"First observation        : {pd.Timestamp(dates[0]).date()}")
print(f"Last observation         : {pd.Timestamp(dates[-1]).date()}")
print(f"Mean daily return        : {returns.mean():.8f}")
print(f"Daily return SD          : {returns.std(ddof=1):.8f}")
print(f"Mean return (%)          : {X.mean():.6f}")
print(f"Daily SD (%)             : {X.std(ddof=1):.6f}")


# ============================================================
# 3. MODEL ESTIMATION FUNCTION
# ============================================================

def estimate_best_hmm(n_states, X):

    best_model = None
    best_log_likelihood = -np.inf
    best_seed = None

    print(
        f"\nEstimating {n_states}-state Gaussian HMM "
        f"using {N_INITIALISATIONS} initialisations..."
    )

    for i in range(N_INITIALISATIONS):

        seed = RANDOM_SEED + i

        try:
            model = GaussianHMM(
                n_components=n_states,
                covariance_type="diag",
                n_iter=MAX_ITERATIONS,
                tol=CONVERGENCE_TOLERANCE,
                min_covar=1e-4,
                random_state=seed
            )

            model.fit(X)

            log_likelihood = model.score(X)

            if (
                np.isfinite(log_likelihood)
                and log_likelihood > best_log_likelihood
            ):
                best_log_likelihood = log_likelihood
                best_model = model
                best_seed = seed

        except Exception as error:
            print(
                f"Initialisation {i + 1} failed: {error}"
            )

    if best_model is None:
        raise RuntimeError(
            f"All {n_states}-state HMM initialisations failed."
        )

    print(f"Best random seed       : {best_seed}")
    print(f"Best log-likelihood    : {best_log_likelihood:.6f}")
    print(f"Converged              : {best_model.monitor_.converged}")
    print(f"Iterations             : {best_model.monitor_.iter}")

    return best_model


# ============================================================
# 4. FIT TWO-STATE AND THREE-STATE MODELS
# ============================================================

model_2 = estimate_best_hmm(2, X)

model_3 = estimate_best_hmm(3, X)


# ============================================================
# 5. MODEL INFORMATION CRITERIA
# ============================================================

def calculate_information_criteria(model, X):

    n = len(X)
    k = model.n_components

    log_likelihood = model.score(X)

    # Parameters:
    # Initial probabilities: k - 1
    # Transition probabilities: k * (k - 1)
    # Means: k
    # Variances: k

    n_parameters = (
        (k - 1)
        + k * (k - 1)
        + 2 * k
    )

    aic = 2 * n_parameters - 2 * log_likelihood

    bic = (
        np.log(n) * n_parameters
        - 2 * log_likelihood
    )

    return {
        "Number_of_States": k,
        "Log_Likelihood": log_likelihood,
        "Number_of_Parameters": n_parameters,
        "AIC": aic,
        "BIC": bic,
        "Converged": model.monitor_.converged,
        "Iterations": model.monitor_.iter
    }


comparison = pd.DataFrame([
    calculate_information_criteria(model_2, X),
    calculate_information_criteria(model_3, X)
])

comparison.to_csv(
    RESULTS_DIR / "hmm_model_comparison.csv",
    index=False
)

print("\n========== MODEL COMPARISON ==========")
print(comparison.to_string(index=False))


# ============================================================
# 6. EXTRACT REGIME INFORMATION
# ============================================================

def analyse_hmm(model, X, dates, n_states):

    # Extract state-specific means and variances.
    means = model.means_.reshape(n_states)

    covariance_matrix = np.asarray(model.covars_)

    variances = covariance_matrix.reshape(
        n_states, -1
    )[:, 0]

    daily_volatility = np.sqrt(variances)

    # Sort regimes from lowest to highest volatility.
    order = np.argsort(daily_volatility)

    old_to_new = np.empty(n_states, dtype=int)
    old_to_new[order] = np.arange(n_states)

    sorted_means = means[order]
    sorted_volatility = daily_volatility[order]

    sorted_transition = model.transmat_[
        np.ix_(order, order)
    ]

    sorted_startprob = model.startprob_[order]

    # Most likely hidden state (Viterbi decoding).
    original_states = model.predict(X)

    states = old_to_new[original_states]

    # Posterior probability of each hidden state.
    posterior_original = model.predict_proba(X)

    posterior = posterior_original[:, order]

    # Assign descriptive regime names.
    if n_states == 2:
        regime_names = [
            "Low Volatility",
            "High Volatility"
        ]
    else:
        regime_names = [
            "Low Volatility",
            "Medium Volatility",
            "High Volatility"
        ]

    # State-specific parameter table.
    parameter_rows = []

    for j in range(n_states):

        probability_stay = sorted_transition[j, j]

        if probability_stay < 1:
            expected_duration = 1 / (1 - probability_stay)
        else:
            expected_duration = np.inf

        parameter_rows.append({
            "State": j,
            "Regime": regime_names[j],
            "Mean_Daily_Return_Percent": sorted_means[j],
            "Daily_Volatility_Percent": sorted_volatility[j],
            "Annualised_Volatility_Percent":
                sorted_volatility[j] * np.sqrt(TRADING_DAYS),
            "Annualised_Mean_Return_Percent":
                sorted_means[j] * TRADING_DAYS,
            "Starting_Probability": sorted_startprob[j],
            "Probability_Staying_In_State": probability_stay,
            "Expected_Duration_Days": expected_duration,
            "Number_of_Observations": np.sum(states == j),
            "Percentage_of_Observations":
                100 * np.mean(states == j)
        })

    parameters = pd.DataFrame(parameter_rows)

    # Regime classification for every trading day.
    classification = pd.DataFrame({
        "Date": dates,
        "Log_Return": X.flatten() / 100,
        "Return_Percent": X.flatten(),
        "State": states,
        "Regime": [regime_names[s] for s in states]
    })

    for j in range(n_states):
        classification[
            f"Probability_{regime_names[j].replace(' ', '_')}"
        ] = posterior[:, j]

    transition_df = pd.DataFrame(
        sorted_transition,
        index=regime_names,
        columns=regime_names
    )

    return (
        parameters,
        classification,
        transition_df,
        regime_names
    )


# ============================================================
# 7. ANALYSE BOTH MODELS
# ============================================================

results = {}

for n_states, model in [(2, model_2), (3, model_3)]:

    (
        parameters,
        classification,
        transition,
        regime_names
    ) = analyse_hmm(model, X, dates, n_states)

    results[n_states] = {
        "parameters": parameters,
        "classification": classification,
        "transition": transition,
        "regime_names": regime_names
    }

    parameters.to_csv(
        RESULTS_DIR / f"hmm_{n_states}_state_parameters.csv",
        index=False
    )

    classification.to_csv(
        RESULTS_DIR / f"hmm_{n_states}_state_classification.csv",
        index=False
    )

    transition.to_csv(
        RESULTS_DIR / f"hmm_{n_states}_state_transition_matrix.csv"
    )

    print(
        f"\n========== {n_states}-STATE REGIME PARAMETERS =========="
    )

    print(parameters.to_string(index=False))

    print(
        f"\n========== {n_states}-STATE TRANSITION MATRIX =========="
    )

    print(transition.to_string())


# ============================================================
# 8. SAVE A TEXT SUMMARY
# ============================================================

summary_file = RESULTS_DIR / "hmm_model_summary.txt"

with open(summary_file, "w", encoding="utf-8") as file:

    file.write("HIDDEN MARKOV MODEL: NIFTY 50\n")
    file.write("=" * 65 + "\n\n")

    file.write(f"Observations: {n_observations}\n")
    file.write(
        f"Sample period: {pd.Timestamp(dates[0]).date()} "
        f"to {pd.Timestamp(dates[-1]).date()}\n\n"
    )

    file.write("MODEL COMPARISON\n")
    file.write("-" * 65 + "\n")
    file.write(comparison.to_string(index=False))

    for n_states in [2, 3]:

        file.write(
            f"\n\n{n_states}-STATE MODEL PARAMETERS\n"
        )
        file.write("-" * 65 + "\n")
        file.write(
            results[n_states]["parameters"].to_string(index=False)
        )

        file.write(
            f"\n\n{n_states}-STATE TRANSITION MATRIX\n"
        )
        file.write(
            results[n_states]["transition"].to_string()
        )

print("\n========== OUTPUT FILES ==========")

print("\nResults:")
for file in sorted(RESULTS_DIR.glob("*")):
    print(file.name)

print("\nHMM estimation completed successfully.")
