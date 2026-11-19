\# Random Walk vs Brownian Motion vs Real Stock Markets



\## A Statistical and Machine Learning Investigation Using NIFTY 50



\### 1. Project Overview



This project investigates the relationship between classical stochastic-process models and the empirical behaviour of financial markets. Using historical NIFTY 50 data, it examines whether commonly used theoretical models adequately capture the statistical characteristics of real stock-market movements.



The study compares Random Walk, Brownian Motion, and Geometric Brownian Motion (GBM) frameworks with more flexible models that account for time-varying volatility and market regimes.



The project combines financial mathematics, statistical analysis, econometric modelling, and machine learning.



\### 2. Research Objectives



The main objectives are:



\- To investigate the Random Walk hypothesis in financial time series.

\- To study the properties of Brownian Motion and its relevance to financial markets.

\- To simulate stock-price dynamics using Geometric Brownian Motion.

\- To examine volatility clustering using GARCH models.

\- To compare Gaussian and Student's t innovations in GARCH modelling.

\- To identify different market regimes using Hidden Markov Models.

\- To evaluate the limitations of classical stochastic models when applied to real financial data.



\### 3. Dataset



The empirical analysis uses the NIFTY 50 index.



| Property | Description |

|---|---|

| Financial instrument | NIFTY 50 |

| Data source | Yahoo Finance |

| Ticker | `^NSEI` |

| Frequency | Daily |

| Sample period | 4 January 2010 – 11 September 2026 |

| Price observations | 4,101 |

| Valid daily returns | 4,100 |



Daily logarithmic returns are calculated as:



\\\[

r\_t = \\ln\\left(\\frac{P\_t}{P\_{t-1}}\\right)

\\]



where \\(P\_t\\) denotes the closing index value at time \\(t\\).



\### 4. Methodology and Models



The project implements the following models.



\#### 4.1 Exploratory Data Analysis



\- Price and return visualisation

\- Return distribution analysis

\- Skewness and kurtosis

\- Autocorrelation analysis

\- Statistical diagnostic tests



\#### 4.2 Random Walk



The Random Walk framework is used to investigate the evolution of financial prices through accumulated random increments.



\#### 4.3 Brownian Motion



Standard Brownian Motion is simulated to examine normally distributed independent increments and the resulting distribution of terminal values.



\#### 4.4 Geometric Brownian Motion



GBM models the evolution of positive asset prices using a continuous-time stochastic process.



\#### 4.5 GARCH(1,1)



A Gaussian GARCH(1,1) model is estimated to investigate conditional heteroskedasticity and volatility clustering.



\#### 4.6 Student's t-GARCH



The Student's t innovation specification is compared with the Gaussian GARCH model to examine the role of heavier-tailed return distributions.



\#### 4.7 Hidden Markov Models



Two-state and three-state Gaussian Hidden Markov Models are implemented to identify latent market regimes and examine their associated characteristics.



\### 5. Repository Structure



```text

Finance\_project/

│

├── data/

│   ├── raw/

│   │   └── nifty50.csv

│   └── processed/

│       └── nifty50.csv

│

├── figures/

│   ├── random\_walk/

│   ├── brownian\_motion/

│   ├── gbm/

│   ├── garch/

│   ├── garch\_student\_t/

│   └── hmm/

│

├── results/

│   ├── random\_walk/

│   ├── brownian\_motion/

│   ├── gbm/

│   ├── garch/

│   ├── garch\_student\_t/

│   ├── hmm/

│   ├── eda\_summary.csv

│   └── statistical\_tests.csv

│

├── src/

│   ├── download\_data.py

│   ├── check\_data.py

│   ├── preprocess\_data.py

│   ├── verify\_processed\_data.py

│   ├── eda.py

│   ├── statistical\_tests.py

│   ├── random\_walk.py

│   ├── brownian\_motion.py

│   ├── gbm.py

│   ├── garch.py

│   ├── garch\_student\_t.py

│   ├── hmm\_regime\_detection.py

│   └── hmm\_visualization.py

│

├── notebooks/

├── README.md

├── requirements.txt

└── .gitignore

```



\### 6. Installation



Clone the repository:



```bash

git clone <YOUR\_REPOSITORY\_URL>

cd Finance\_project

```



Create a virtual environment:



```bash

python -m venv .venv

```



Activate it on Windows:



```bash

.venv\\Scripts\\activate

```



Install the required packages:



```bash

python -m pip install --upgrade pip

pip install -r requirements.txt

```



\### 7. Execution



The main Python scripts are located in the `src/` directory.



The general workflow is:



1\. Download and inspect the historical data.

2\. Preprocess and verify the dataset.

3\. Perform exploratory data analysis and statistical testing.

4\. Implement Random Walk, Brownian Motion, and GBM simulations.

5\. Estimate Gaussian GARCH and Student's t-GARCH models.

6\. Estimate Hidden Markov Models and visualise market regimes.



For example:



```bash

python src/eda.py

python src/random\_walk.py

python src/brownian\_motion.py

python src/gbm.py

python src/garch.py

python src/garch\_student\_t.py

python src/hmm\_regime\_detection.py

python src/hmm\_visualization.py

python src/statistical\_tests.py

```



Run the scripts from the project root directory. Some scripts depend on the processed dataset and outputs generated by earlier steps.



\### 8. Results and Outputs



The project stores its outputs in two main directories:



\- `figures/`: Visualisations of simulations, return distributions, conditional volatility, and estimated market regimes.

\- `results/`: Model parameters, statistical summaries, diagnostic results, model comparisons, and regime classifications.



These outputs allow the theoretical models to be examined alongside the empirical characteristics of the NIFTY 50 index.



\### 9. Limitations



\- The empirical investigation focuses on a single market index.

\- Stochastic simulations depend on estimated historical parameters.

\- Gaussian models may not adequately represent extreme market movements.

\- GARCH specifications capture conditional volatility but do not explain every source of market variation.

\- Hidden Markov Model regime classifications are model-dependent and may be sensitive to the selected number of states and estimation sample.

\- The analysis should not be interpreted as a guaranteed stock-market forecasting or investment strategy.



\### 10. Future Scope



Potential extensions include:



\- Out-of-sample model evaluation.

\- Alternative volatility models such as EGARCH and GJR-GARCH.

\- Comparison across multiple financial indices.

\- Advanced regime-switching volatility models.

\- Machine learning-based volatility forecasting.

\- Risk measurement using Value at Risk and Expected Shortfall.



\### 11. Technologies Used



\- Python

\- NumPy

\- Pandas

\- Matplotlib

\- SciPy

\- Statsmodels

\- ARCH

\- hmmlearn

\- yfinance



\---



\*\*Project Category:\*\* Financial Econometrics | Stochastic Processes | Statistical Modelling | Machine Learning



\*\*Primary Application:\*\* Empirical analysis of NIFTY 50 financial time series.
