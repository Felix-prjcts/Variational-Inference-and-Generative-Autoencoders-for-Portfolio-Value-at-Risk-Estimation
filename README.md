# Variational autoencoders for portfolio Value-at-Risk

Félix Bouvier, Mathis Rivallan — Toulouse School of Economics, Magistère Économiste Statisticien (2A), 2025/2026.

We train a variational autoencoder (VAE) on the daily log-returns of a 60-asset portfolio, sample synthetic market scenarios from its latent space, and use them to estimate the portfolio's VaR and CVaR. The estimates are compared with the historical and Gaussian approaches and backtested with the Kupiec test.

**The full write-up is in [`VAE_case_study.pdf`](VAE_case_study.pdf)** (in French). It covers the theory (ELBO, reparameterization trick, KL term), the model and data choices, and the detailed discussion of the results. Read it first; this README only gives a summary.

## Main results

Setup: 60 assets (equity indices, sector and country ETFs, commodities, bonds and rates, USD index), equal weights, daily data from November 2004 to June 2026 (4,448 returns). VAE with 2×128 hidden layers, latent dimension 18, β = 1, 120 epochs; 10,000 generated scenarios.

**Portfolio returns, generated vs historical**

| | VAE | Historical |
|---|---|---|
| Std. deviation | 0.85% | 0.96% |
| Skewness | -0.40 | -0.63 |
| Excess kurtosis | 2.69 | 13.35 |
| Min / max | -5.20% / 4.80% | -8.95% / 8.59% |

The VAE gets the volatility and the sign of the skewness right but produces much thinner tails. Asset by asset, the two-sample KS test rejects equality of distributions for 57 of the 60 assets.

**VaR and CVaR (daily returns)**

| Level | VAE VaR | VAE CVaR | Historical VaR | Historical CVaR |
|---|---|---|---|---|
| 95% | -1.46% | -2.12% | -1.38% | -2.43% |
| 99% | -2.53% | -3.10% | -2.85% | -4.48% |

**Kupiec backtest** (VaR from the VAE scenarios, tested on the 4,448 realised returns, α = 5%)

| Level | Violations (expected) | p-value | Result |
|---|---|---|---|
| 95% | 207 (222) | 0.284 | not rejected |
| 99% | 67 (44) | 0.0016 | rejected |

In short: the VAE-based VaR is well calibrated at 95% but underestimates extreme risk at 99%, which is consistent with the kurtosis gap and with the known limits of a Gaussian prior/decoder for fat-tailed data. A heavier-tailed decoder (e.g. Student-t) would be the natural next step. The Gaussian parametric VaR is rejected even in-sample, at both levels, so the normality assumption is inadequate for this portfolio regardless of the VAE.

## Repository

```text
notebooks/
  01_data_exploration.ipynb    download prices, returns, descriptive stats
  02_vae_training.ipynb        train the VAE, latent space (UMAP)
  03_scenario_generation.ipynb sample scenarios, compare with history (KS tests)
  04_var_calculation.ipynb     VaR / CVaR, Kupiec backtest
src/
  data_loader.py               Yahoo Finance download and returns
  vae_model.py                 VAE model, loss and training loop
  var_calculator.py            VaR, CVaR, Kupiec test
  utils.py                     configuration (tickers, dates, hyperparameters)
VAE_case_study.pdf             report
```

## Running it

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
mkdir -p results
cd notebooks && jupyter notebook
```

Run the notebooks in order: each one reads what the previous one wrote to `data/` and `results/`. Prices are downloaded from Yahoo Finance, so a network connection is needed and the numbers can differ slightly from the report if the data source changes. Tickers, dates and hyperparameters are set in `PortfolioConfig` in `src/utils.py`.
