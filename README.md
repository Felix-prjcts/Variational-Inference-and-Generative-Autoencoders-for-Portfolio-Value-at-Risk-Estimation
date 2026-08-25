# VAE portfolio VaR project

This repository builds synthetic market scenarios for a multi-asset portfolio using a variational autoencoder and then evaluates risk with VaR-style metrics.

The workflow is organized around a small set of notebooks:

1. load and inspect market data
2. train the VAE on portfolio returns
3. generate scenarios from the learned latent space
4. compute risk measures from the generated outcomes

## Project structure

```text
.
├── data/
│   └── ...
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_vae_training.ipynb
│   ├── 03_scenario_generation.ipynb
│   └── 04_var_calculation.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── vae_model.py
│   ├── var_calculator.py
│   └── utils.py
├── requirements.txt
├── QUICKSTART.md
├── explication_methodes_var.txt
└── README.md
```

## Setup

Create a Python environment and install the project dependencies:

```bash
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you are using Conda instead, the project can also be run from a standard environment with the same package requirements.

## Running the project

Open the notebooks in order:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

Then continue with:

- 02_vae_training.ipynb
- 03_scenario_generation.ipynb
- 04_var_calculation.ipynb

Each notebook is meant to be run sequentially so the data, trained model, and scenario outputs are available for the next step.

## Notes on the model

The VAE is trained on portfolio return data and used to generate plausible market scenarios. Those generated scenarios are then fed into a VaR calculation workflow, which compares different risk estimators such as historical VaR and parametric VaR.

The exact configuration can be adjusted in the project utilities, especially the portfolio definition, latent dimension, training parameters, and scenario count.

### Environment issues

If the virtual environment is incomplete or the Python executable is missing from `.venv\Scripts`, recreate it:

```bash
Remove-Item -Recurse -Force .venv
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Data or notebook issues

If the data import fails, check that the required packages are installed and that the notebook is being run from the project root. In some cases, the issue is caused by a stale environment or mismatched package versions.

### Training or memory issues

If training is slow or crashes due to memory pressure, reduce the number of training epochs or generated scenarios in the configuration values used by the project.

