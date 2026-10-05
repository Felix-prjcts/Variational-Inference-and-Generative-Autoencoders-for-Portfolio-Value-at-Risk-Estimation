# VAE portfolio project quick start

## Project layout

```text
VAE project/
├── src/
│   ├── data_loader.py
│   ├── vae_model.py
│   ├── var_calculator.py
│   └── utils.py
├── notebooks/
├── data/
├── results/
├── environment.yml
├── requirements.txt
├── README.md
└── QUICKSTART.md
```

## Environment setup

Create and activate an environment:

```bash
conda env create -f environment.yml
conda activate vae-portfolio
```

If you prefer a standard virtual environment:

```bash
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Verify the installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Notebook workflow

Run the notebooks in order:

1. 01_data_exploration.ipynb
   - load market data
   - compute returns and summary statistics
   - inspect correlations and price trends

2. 02_vae_training.ipynb
   - train the VAE on historical returns
   - monitor training loss
   - save the model output

3. 03_scenario_generation.ipynb
   - generate synthetic scenarios from the latent distribution
   - compare generated data against historical data

4. 04_var_calculation.ipynb
   - compute risk metrics
   - compare VaR methods and summarize the results

Start with:

```bash
cd notebooks
jupyter notebook 01_data_exploration.ipynb
```

## Configuration

The core portfolio and training settings are in [src/utils.py](src/utils.py), inside `PortfolioConfig`.

```python
self.indices = [...]
self.weights = np.ones(len(self.indices)) / len(self.indices)
self.latent_dim = 18
self.hidden_dim = 128
self.n_epochs = 120
self.batch_size = 64
self.n_scenarios = 10000
```

Adjust these values to change the asset universe, portfolio weights, model capacity, or number of generated scenarios.

## Expected outputs

The project writes outputs under the `results/` directory, including:

- training plots
- scenario comparisons
- portfolio statistics
- VaR summary tables
- saved model weights

## Understanding the results

The main risk metrics are based on the distribution of simulated portfolio losses.

- Historical VaR: computed from empirical loss quantiles
- Parametric VaR: based on the assumed distribution of returns
- CVaR: tail expectation beyond the VaR threshold

## Troubleshooting

### ModuleNotFoundError

```bash
conda activate vae-portfolio
python -c "import torch, pandas, sklearn, yfinance"
```

### Data download issues

- check the network connection
- verify the ticker list and date range
- test the data source manually if needed

### Memory or training issues

- lower `n_scenarios`
- lower `batch_size`
- reduce the number of epochs
- use CPU if GPU memory is limited

### Slow training

- reduce `latent_dim`
- lower `n_epochs`
- switch to a machine with better GPU support if available

## References

- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes.
- standard VaR and expected-shortfall methodology in portfolio risk analysis
- PyTorch and pandas documentation for the underlying libraries

## Notes

This project is meant to be used as a practical notebook-based workflow for portfolio scenario generation and risk analysis. The exact outputs depend on the data source, portfolio configuration, and runtime environment.
