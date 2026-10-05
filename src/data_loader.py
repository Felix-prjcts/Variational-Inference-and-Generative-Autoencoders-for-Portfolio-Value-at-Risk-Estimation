"""Download and prepare daily index prices from Yahoo Finance."""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class PortfolioDataLoader:

    def __init__(self, indices, start_date, end_date):
        self.indices = indices
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.returns = None
        self.scaler = StandardScaler()

    def fetch_data(self):
        """Return a DataFrame of close prices, one column per ticker."""
        df = yf.download(
            tickers=self.indices,
            start=self.start_date,
            end=self.end_date,
            progress=False,
            auto_adjust=False
        )

        if 'Close' in df.columns.levels[0] if isinstance(df.columns, pd.MultiIndex) else 'Close' in df.columns:
            self.data = df['Close']
        elif 'Adj Close' in df.columns.levels[0] if isinstance(df.columns, pd.MultiIndex) else 'Adj Close' in df.columns:
            self.data = df['Adj Close']
        else:
            raise KeyError("Neither 'Close' nor 'Adj Close' found in downloaded data")

        # with a single ticker yfinance may return a Series
        if isinstance(self.data, pd.Series):
            self.data = self.data.to_frame(name=self.indices[0])
        elif len(self.indices) == 1 and isinstance(self.data, pd.DataFrame):
            self.data.columns = self.indices

        self.data = self.data.dropna()
        return self.data

    def calculate_returns(self, method='log'):
        """method: 'log' or 'simple'."""
        if self.data is None:
            raise ValueError("Data not loaded. Call fetch_data() first.")

        if method == 'log':
            self.returns = np.log(self.data / self.data.shift(1)).dropna()
        else:
            self.returns = self.data.pct_change().dropna()

        return self.returns

    def normalize_returns(self, fit=True):
        if self.returns is None:
            raise ValueError("Returns not calculated. Call calculate_returns() first.")

        if fit:
            returns_normalized = self.scaler.fit_transform(self.returns)
        else:
            returns_normalized = self.scaler.transform(self.returns)

        return returns_normalized

    def get_data_summary(self):
        if self.returns is None:
            raise ValueError("Returns not calculated.")

        summary = {
            'mean': self.returns.mean(),
            'std': self.returns.std(),
            'min': self.returns.min(),
            'max': self.returns.max(),
            'sharpe': self.returns.mean() / self.returns.std() * np.sqrt(252),  # annualised
        }
        return summary


def create_train_test_split(data, test_size=0.2):
    """Chronological split (no shuffling)."""
    split_idx = int(len(data) * (1 - test_size))
    X_train = data[:split_idx]
    X_test = data[split_idx:]
    return X_train, X_test
