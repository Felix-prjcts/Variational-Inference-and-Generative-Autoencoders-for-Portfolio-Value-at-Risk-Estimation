"""
Data loader for financial indices 
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class PortfolioDataLoader:
    """Load and process portfolio data from Yahoo Finance"""
    
    def __init__(self, indices, start_date, end_date):
        """
        Initialize the data loader
        
        Args:
            indices: list of ticker symbols (e.g., ['^GSPC', '^STOXX50E'])
            start_date: start date string (YYYY-MM-DD)
            end_date: end date string (YYYY-MM-DD)
        """
        self.indices = indices
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.returns = None
        self.scaler = StandardScaler()
        
    def fetch_data(self):
        """
        Fetch historical price data from Yahoo Finance
        
        Returns:
            DataFrame with close prices used for return calculations
        """
        print(f"Fetching data for {len(self.indices)} indices...")
        
        try:
            # Download all tickers at once (much faster and cleaner)
            df = yf.download(
                tickers=self.indices,
                start=self.start_date,
                end=self.end_date,
                progress=False,
                auto_adjust=False
            )
            
            # Handle the column selection safely based on available columns
            if 'Close' in df.columns.levels[0] if isinstance(df.columns, pd.MultiIndex) else 'Close' in df.columns:
                self.data = df['Close']
            elif 'Adj Close' in df.columns.levels[0] if isinstance(df.columns, pd.MultiIndex) else 'Adj Close' in df.columns:
                self.data = df['Adj Close']
            else:
                raise KeyError("Neither 'Close' nor 'Adj Close' found in downloaded data")
            
            # If only one ticker was requested, yf.download returns a Series or single-column DF. 
            # We ensure self.data is always a DataFrame with tickers as columns.
            if isinstance(self.data, pd.Series):
                self.data = self.data.to_frame(name=self.indices[0])
            elif len(self.indices) == 1 and isinstance(self.data, pd.DataFrame):
                self.data.columns = self.indices

            self.data = self.data.dropna()
            print(f"✓ Successfully loaded data. Shape: {self.data.shape}")
            return self.data

        except Exception as e:
            print(f"✗ Error during data fetching: {e}")
            raise e
    
    def calculate_returns(self, method='log'):
        """
        Calculate returns from prices
        
        Args:
            method: 'log' for log returns or 'simple' for simple returns
            
        Returns:
            DataFrame with returns
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call fetch_data() first.")
        
        if method == 'log':
            self.returns = np.log(self.data / self.data.shift(1)).dropna()
        else:
            self.returns = self.data.pct_change().dropna()
        
        print(f"Returns shape: {self.returns.shape}")
        print(f"Mean returns:\n{self.returns.mean()}\n")
        
        return self.returns
    
    def normalize_returns(self, fit=True):
        """
        Normalize returns using StandardScaler
        
        Args:
            fit: if True, fit scaler on data; if False, use fitted scaler
            
        Returns:
            normalized returns (numpy array)
        """
        if self.returns is None:
            raise ValueError("Returns not calculated. Call calculate_returns() first.")
        
        if fit:
            returns_normalized = self.scaler.fit_transform(self.returns)
        else:
            returns_normalized = self.scaler.transform(self.returns)
        
        print(f"Normalized returns shape: {returns_normalized.shape}")
        return returns_normalized
    
    def get_data_summary(self):
        """Get summary statistics of returns"""
        if self.returns is None:
            raise ValueError("Returns not calculated.")
        
        summary = {
            'mean': self.returns.mean(),
            'std': self.returns.std(),
            'min': self.returns.min(),
            'max': self.returns.max(),
            'sharpe': self.returns.mean() / self.returns.std() * np.sqrt(252),  # Annualized
        }
        return summary


def create_train_test_split(data, test_size=0.2):
    """
    Split data into train and test sets
    
    Args:
        data: numpy array of shape (n_samples, n_features)
        test_size: proportion of test data (0-1)
        
    Returns:
        X_train, X_test
    """
    split_idx = int(len(data) * (1 - test_size))
    X_train = data[:split_idx]
    X_test = data[split_idx:]
    
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    return X_train, X_test
