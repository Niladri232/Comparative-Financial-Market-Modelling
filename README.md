# Comparative Financial Market Modelling

### A Statistical, Econometric and Machine Learning Investigation of Financial Markets

---

## 1. Project Overview

Financial markets exhibit complex statistical behaviour characterised by uncertainty, volatility clustering, non-normal return distributions and changing market conditions. Classical stochastic-process models provide a mathematical foundation for understanding financial price movements, but their assumptions may not fully capture the empirical characteristics of real-world financial markets.

**Comparative Financial Market Modelling** investigates the relationship between theoretical stochastic-process models and the observed behaviour of financial time series.

The initial empirical investigation focuses on the **NIFTY 50 index**, examining the performance and limitations of different approaches to financial market modelling.

The project integrates concepts from financial mathematics, probability theory, statistical analysis, econometrics and machine learning.

The current implementation includes:

- Random Walk simulations
- Standard Brownian Motion
- Geometric Brownian Motion (GBM)
- Gaussian GARCH(1,1)
- Student's t-GARCH
- Gaussian Hidden Markov Models (HMM)
- Exploratory Data Analysis and statistical diagnostics

The broader objective is to develop a comparative framework that can eventually be extended to multiple financial indices, sectors and individual securities.

## 2. Research Motivation and Questions

Financial asset prices are frequently modelled using stochastic processes. However, real financial returns often exhibit characteristics such as heavy tails, volatility clustering and structural changes that challenge the assumptions of simpler models.

This project addresses the following research questions:

1. How well do Random Walk and Brownian Motion frameworks represent the behaviour of financial markets?
2. How effectively does Geometric Brownian Motion reproduce the dynamics of observed financial prices?
3. Does GARCH modelling capture the volatility clustering present in financial returns?
4. Does a Student's t innovation specification provide a more suitable representation of extreme return observations than Gaussian innovations?
5. Can Hidden Markov Models identify distinct latent market regimes?
6. What are the limitations of these models when applied to empirical financial time series?

## 3. Research Objectives

The main objectives are:

- To investigate the Random Walk hypothesis in financial time series.
- To study the mathematical properties of Brownian Motion and its relevance to financial markets.
- To simulate financial price dynamics using Geometric Brownian Motion.
- To examine volatility clustering through GARCH modelling.
- To compare Gaussian and Student's t innovation specifications.
- To identify latent market regimes using Hidden Markov Models.
- To evaluate the limitations of classical stochastic and econometric models when applied to real financial data.
- To establish a foundation for future comparative financial modelling using machine learning techniques.

## 4. Dataset and Data Description

### 4.1 Data Source

The initial empirical investigation uses historical daily data for the NIFTY 50 index, obtained through Yahoo Finance.

| Property | Description |
|---|---|
| Financial instrument | NIFTY 50 |
| Data source | Yahoo Finance |
| Ticker | `^NSEI` |
| Frequency | Daily |
| Sample period | 4 January 2010 – 11 September 2026 |
| Price observations | 4,101 |
| Valid daily returns | 4,100 |

### 4.2 Return Calculation

Daily logarithmic returns are calculated from consecutive closing index values:

\[
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
\]

where:

- \(P_t\) represents the closing index value at time \(t\).
- \(P_{t-1}\) represents the previous trading day's closing index value.
- \(r_t\) represents the logarithmic return.

Logarithmic returns are used throughout the empirical analysis to investigate the statistical properties of financial market movements.

## 5. Methodology and Models

The project implements several stochastic, statistical and econometric approaches.

### 5.1 Exploratory Data Analysis (EDA)

Exploratory Data Analysis is conducted to understand the characteristics of the NIFTY 50 price series and its corresponding returns.

The analysis includes:

- Historical price visualisation.
- Daily return visualisation.
- Return distribution analysis.
- Skewness and kurtosis.
- Autocorrelation analysis.
- Statistical diagnostic tests.

### 5.2 Random Walk

The Random Walk framework represents price movements as accumulated random increments.

A basic representation is:

\[
P_t = P_{t-1} + \epsilon_t
\]

where \(\epsilon_t\) represents a random increment.

The model is used to investigate the evolution of financial prices under a random movement assumption and to generate simulated price paths.

### 5.3 Brownian Motion

Standard Brownian Motion is a continuous-time stochastic process characterised by:

- \(W_0=0\).
- Independent increments.
- Normally distributed increments.
- Continuous sample paths.

Its increment structure is:

\[
W_{t+\Delta t}-W_t
\sim N(0,\Delta t)
\]

The simulation examines the properties of Brownian Motion and the distribution of its resulting terminal values.

### 5.4 Geometric Brownian Motion (GBM)

Geometric Brownian Motion is commonly used to represent the evolution of positive financial asset prices.

It is defined by the stochastic differential equation:

\[
dS_t = \mu S_t\,dt+\sigma S_t\,dW_t
\]

where:

- \(S_t\) represents the asset price.
- \(\mu\) represents the drift parameter.
- \(\sigma\) represents volatility.
- \(W_t\) represents standard Brownian Motion.

The GBM model is implemented to simulate financial price trajectories using historically estimated return parameters.

### 5.5 Gaussian GARCH(1,1)

Financial returns frequently exhibit volatility clustering, where periods of high volatility tend to be followed by further periods of high volatility.

The GARCH(1,1) model captures this behaviour through a time-varying conditional variance.

The model is specified as:

\[
r_t=\mu+\epsilon_t
\]

\[
\epsilon_t=\sigma_t z_t
\]

\[
\sigma_t^2=
\omega+\alpha\epsilon_{t-1}^2+
\beta\sigma_{t-1}^2
\]

where:

- \(\omega\) is the variance intercept.
- \(\alpha\) captures the effect of past squared shocks.
- \(\beta\) captures volatility persistence.
- \(\sigma_t^2\) represents conditional variance.

A Gaussian GARCH(1,1) specification is estimated to investigate conditional heteroskedasticity in financial returns.

### 5.6 Student's t-GARCH

Financial return distributions may exhibit heavier tails than the normal distribution.

To investigate this characteristic, the project implements a GARCH model with Student's t innovations.

This specification allows the conditional return distribution to accommodate heavier tails and provides a comparison with the Gaussian GARCH model.

The comparison focuses on the role of the assumed innovation distribution in modelling financial return behaviour.

### 5.7 Hidden Markov Models (HMM)

Financial markets may move between different latent regimes, such as relatively calm and highly volatile periods.

Hidden Markov Models provide a probabilistic framework for identifying unobserved states underlying an observed time series.

The project implements:

- Two-state Gaussian Hidden Markov Model.
- Three-state Gaussian Hidden Markov Model.

The models are used to identify latent market regimes and examine their associated characteristics.

The resulting classifications are model-based estimates. Full-sample regime classifications should be interpreted as retrospective analysis rather than guaranteed real-time predictions.

## 6. Experimental Results and Outputs

The project generates numerical results, model estimates, statistical summaries and visualisations.

The outputs are organised into two main directories:

**Figures (`figures/`)**

Contains visualisations associated with:

- Random Walk simulations.
- Brownian Motion simulations.
- Geometric Brownian Motion.
- Gaussian GARCH.
- Student's t-GARCH.
- Hidden Markov Models.

**Results (`results/`)**

Contains:

- Model-specific numerical outputs.
- Simulation results.
- Statistical summaries.
- Model estimates.
- Market regime classifications.
- EDA summary statistics.
- Statistical diagnostic test results.

The numerical outputs and figures can be used to examine the relationship between theoretical model assumptions and the empirical characteristics of the NIFTY 50 index.

## 7. Repository Structure

```text
Comparative-Financial-Market-Modelling/
│
├── data/
│   ├── raw/
│   │   └── nifty50.csv
│   │
│   └── processed/
│       └── nifty50.csv
│
├── figures/
│   ├── random_walk/
│   ├── brownian_motion/
│   ├── gbm/
│   ├── garch/
│   ├── garch_student_t/
│   └── hmm/
│
├── results/
│   ├── random_walk/
│   ├── brownian_motion/
│   ├── gbm/
│   ├── garch/
│   ├── garch_student_t/
│   ├── hmm/
│   ├── eda_summary.csv
│   └── statistical_tests.csv
│
├── src/
│   ├── download_data.py
│   ├── check_data.py
│   ├── preprocess_data.py
│   ├── verify_processed_data.py
│   ├── eda.py
│   ├── statistical_tests.py
│   ├── random_walk.py
│   ├── brownian_motion.py
│   ├── gbm.py
│   ├── garch.py
│   ├── garch_student_t.py
│   ├── hmm_regime_detection.py
│   └── hmm_visualization.py
│
├── notebooks/
│
├── .gitignore
├── README.md
└── requirements.txt
```

## 8. Installation and Setup

### 8.1 Clone the Repository

```bash
git clone https://github.com/Niladri232/Comparative-Financial-Market-Modelling.git

cd Comparative-Financial-Market-Modelling
```

### 8.2 Create a Virtual Environment

```bash
python -m venv .venv
```

### 8.3 Activate the Virtual Environment

**Windows:**

```cmd
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
source .venv/bin/activate
```

### 8.4 Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## 9. Execution Workflow

All Python scripts are located in the `src/` directory.

Run the commands from the repository's root directory.

### Step 1: Data Collection and Preparation

```bash
python src/download_data.py
python src/check_data.py
python src/preprocess_data.py
python src/verify_processed_data.py
```

### Step 2: Exploratory and Statistical Analysis

```bash
python src/eda.py
python src/statistical_tests.py
```

### Step 3: Stochastic Process Simulations

```bash
python src/random_walk.py
python src/brownian_motion.py
python src/gbm.py
```

### Step 4: Volatility Modelling

```bash
python src/garch.py
python src/garch_student_t.py
```

### Step 5: Market Regime Detection

```bash
python src/hmm_regime_detection.py
python src/hmm_visualization.py
```

**Note:** Some scripts depend on processed datasets or output files generated by earlier scripts. Follow the workflow above and ensure the required input files exist before running individual models. Data downloading may also require an active internet connection.

## 10. Limitations

Despite providing a framework for analysing financial market behaviour, the current implementation has several limitations:

- The initial empirical investigation focuses on a single financial index, NIFTY 50.
- Stochastic simulations depend on historically estimated parameters.
- Standard Brownian Motion and GBM rely on assumptions that may not fully reflect actual market dynamics.
- Gaussian models may not adequately represent extreme financial return observations.
- GARCH models capture conditional volatility but do not explain every source of market variation.
- Hidden Markov Model classifications depend on the selected number of states, model specification and estimation sample.
- Full-sample regime classifications are retrospective and should not automatically be interpreted as real-time forecasts.
- The current implementation does not establish a guaranteed predictive or profitable investment strategy.

## 11. Future Research Directions

The project is designed to support further development in financial econometrics and data-driven financial modelling.

Potential extensions include:

### 11.1 Comparative Financial Market Analysis

- Extend the empirical analysis to multiple financial indices.
- Investigate sector-specific indices.
- Include individual stocks from different sectors.
- Compare the statistical properties of different financial instruments.

### 11.2 Advanced Econometric Modelling

- EGARCH and GJR-GARCH models.
- Advanced volatility forecasting.
- Regime-switching volatility models.
- Out-of-sample model evaluation.

### 11.3 Machine Learning and Deep Learning

- Random Forest.
- Support Vector Machines.
- XGBoost.
- Long Short-Term Memory Networks (LSTM).
- Gated Recurrent Units (GRU).
- Transformer-based time-series models.

### 11.4 Financial Risk Analysis

- Value at Risk (VaR).
- Expected Shortfall (ES).
- Portfolio risk analysis.
- Volatility forecasting.
- Market regime prediction.

These are planned research extensions and are not claimed as completed implementations.

## 12. Technologies and Libraries

The project uses the following technologies:

| Technology | Application |
|---|---|
| Python | Programming and modelling |
| NumPy | Numerical computation |
| Pandas | Data manipulation |
| Matplotlib | Data visualisation |
| SciPy | Statistical computation |
| Statsmodels | Statistical analysis |
| ARCH | GARCH modelling |
| hmmlearn | Hidden Markov Models |
| yfinance | Financial data acquisition |

## 13. Project Status

**Current implementation:**

- NIFTY 50 historical data collection and preprocessing.
- Exploratory Data Analysis.
- Statistical diagnostic testing.
- Random Walk simulation.
- Brownian Motion simulation.
- Geometric Brownian Motion simulation.
- Gaussian GARCH(1,1) estimation.
- Student's t-GARCH estimation.
- Two-state and three-state Gaussian HMM implementation.
- Numerical results and visualisations.

**Research direction:** Expansion towards comparative financial market modelling, advanced econometric methods and machine learning-based financial analysis.

## 14. Author and Project Information

**Project Name:** Comparative Financial Market Modelling

**Research Area:** Financial Economics, Stochastic Processes, Financial Econometrics, Statistical Modelling and Machine Learning

**Primary Empirical Application:** NIFTY 50 Financial Time Series

**Data Source:** Yahoo Finance

**GitHub Repository:**  
https://github.com/Niladri232/Comparative-Financial-Market-Modelling

---

*This project is developed as an academic and research-oriented investigation into the statistical and mathematical modelling of financial markets. It is intended for educational and research purposes and should not be considered financial or investment advice.*
