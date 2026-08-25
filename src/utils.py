"""
Utility functions for the VAE Portfolio project
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch


def calculate_log_returns(prices):
    """
    Calculate log returns from price series
    
    Args:
        prices: numpy array or pandas Series of prices
        
    Returns:
        log returns
    """
    if isinstance(prices, pd.Series):
        prices = prices.values
    return np.diff(np.log(prices))


def normalize_data(data, scaler=None, fit=False):
    """
    Normalize data using StandardScaler
    
    Args:
        data: numpy array (n_samples, n_features)
        scaler: StandardScaler object or None
        fit: if True, fit the scaler on the data
        
    Returns:
        normalized data, scaler object
    """
    if scaler is None:
        scaler = StandardScaler()
        if fit:
            data_normalized = scaler.fit_transform(data)
        else:
            data_normalized = scaler.transform(data)
    else:
        data_normalized = scaler.transform(data)
    
    return data_normalized, scaler


def denormalize_data(data_normalized, scaler):
    """
    Denormalize data back to original scale
    
    Args:
        data_normalized: normalized data
        scaler: StandardScaler object
        
    Returns:
        denormalized data
    """
    return scaler.inverse_transform(data_normalized)


def numpy_to_torch(data, device='cpu'):
    """
    Convert numpy array to torch tensor
    
    Args:
        data: numpy array
        device: 'cpu' or 'cuda'
        
    Returns:
        torch tensor
    """
    return torch.FloatTensor(data).to(device)


def torch_to_numpy(tensor):
    """
    Convert torch tensor to numpy array
    
    Args:
        tensor: torch tensor
        
    Returns:
        numpy array
    """
    return tensor.cpu().detach().numpy()


class PortfolioConfig:
    """Configuration class for portfolio parameters"""
    
    def __init__(self):
        self.indices = [
            '^GSPC', '^DJI', '^IXIC', '^FTSE', '^FCHI', '^GDAXI', '^SSMI', '^N225', '^HSI', '^AEX',
            'SPY', 'QQQ', 'DIA', 'IWM', 'IVV', 'EFA', 'EEM', 'VTI',
            'XLF', 'XLK', 'XLE', 'XLY', 'XLP', 'XLI', 'XLB', 'XLV', 'XLU', 'IYR', 'VOX',
            'EWJ', 'EWU', 'EWG', 'EWQ', 'EWC', 'EWH', 'EWW', 'EWA', 'EWS', '^BSESN', 'FXI', '^AXJO',
            'GLD', 'GC=F', 'CL=F', 'NG=F', 'KC=F', 'ZC=F', 'ZW=F', 'HG=F', 'ES=F', 'NQ=F',
            'TLT', 'IEF', 'SHY', 'LQD', '^TNX', 'TIP', '^TYX', 'AGG', 'DX-Y.NYB'
        ]
        self.start_date = '2004-01-01'
        self.end_date = '2026-06-01'
        self.weights = np.ones(len(self.indices)) / len(self.indices)
        
        # VAE parameters
        self.latent_dim = 18
        self.hidden_dim = 128
        self.n_epochs = 120
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # VaR parameters
        self.var_levels = [0.95, 0.99]
        self.n_scenarios = 10000
