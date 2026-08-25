"""
Value at Risk (VaR) calculation module
"""
import numpy as np
import pandas as pd
from scipy import stats


def kupiec_test(returns, var_estimate, confidence_level, alpha=0.05):
    """
    Perform the Kupiec POF backtest for a VaR estimate.

    Args:
        returns: array-like of portfolio returns
        var_estimate: VaR estimate used for the test
        confidence_level: confidence level of the VaR estimate (e.g. 0.95)
        alpha: significance level for the test

    Returns:
        Dictionary with exceedance count, rates, LR statistic, p-value, and conclusion
    """
    returns = np.asarray(returns, dtype=float).ravel()
    if len(returns) == 0:
        raise ValueError("returns must not be empty")

    violations = int(np.sum(returns < var_estimate))
    n_observations = len(returns)
    observed_rate = violations / n_observations
    expected_rate = 1 - confidence_level

    p = float(np.clip(expected_rate, 1e-12, 1 - 1e-12))
    x = float(violations)
    n = float(n_observations)
    observed_prob = float(np.clip(x / n, 1e-12, 1 - 1e-12))

    log_num = (n - x) * np.log(1 - p) + x * np.log(p)
    log_den = (n - x) * np.log(1 - observed_prob) + x * np.log(observed_prob)
    lr_statistic = -2 * (log_num - log_den)

    p_value = 1 - stats.chi2.cdf(lr_statistic, df=1)
    critical_value = stats.chi2.ppf(1 - alpha, df=1)
    reject_null = lr_statistic > critical_value

    conclusion = "VaR validée" if not reject_null else "VaR rejetée"

    return {
        'violations': violations,
        'observed_rate': observed_rate,
        'expected_rate': expected_rate,
        'lr_statistic': lr_statistic,
        'p_value': p_value,
        'conclusion': conclusion,
    }


class VaRCalculator:
    """Calculate Value at Risk for portfolio"""
    
    def __init__(self, portfolio_returns, weights):
        """
        Initialize VaR calculator
        
        Args:
            portfolio_returns: numpy array of portfolio returns (n_samples,)
            weights: numpy array of asset weights (n_assets,)
        """
        self.portfolio_returns = portfolio_returns
        self.weights = weights / np.sum(weights)  # Normalize weights
        
    def calculate_portfolio_returns(self, asset_returns):
        """
        Calculate portfolio returns from individual asset returns
        
        Args:
            asset_returns: numpy array of shape (n_samples, n_assets)
            
        Returns:
            portfolio returns of shape (n_samples,)
        """
        return np.dot(asset_returns, self.weights)
    
    def historical_var(self, confidence_level=0.95):
        """
        Calculate historical VaR
        
        Args:
            confidence_level: confidence level (e.g., 0.95, 0.99)
            
        Returns:
            VaR value
        """
        var = np.percentile(self.portfolio_returns, (1 - confidence_level) * 100)
        return var
    
    def parametric_var(self, confidence_level=0.95):
        """
        Calculate parametric (normal) VaR
        
        Args:
            confidence_level: confidence level (e.g., 0.95, 0.99)
            
        Returns:
            VaR value
        """
        mu = np.mean(self.portfolio_returns)
        sigma = np.std(self.portfolio_returns)
        z_score = stats.norm.ppf(1 - confidence_level)
        var = mu + z_score * sigma
        return var
    
    def monte_carlo_var(self, n_scenarios=10000, confidence_level=0.95):
        """
        Monte Carlo VaR (already computed from simulated returns)
        
        Args:
            n_scenarios: number of scenarios (for reference)
            confidence_level: confidence level
            
        Returns:
            VaR value
        """
        # Use historical percentile on the provided returns
        var = np.percentile(self.portfolio_returns, (1 - confidence_level) * 100)
        return var
    
    def cvar_expected_shortfall(self, confidence_level=0.95):
        """
        Calculate Conditional VaR (CVaR) / Expected Shortfall
        
        Args:
            confidence_level: confidence level
            
        Returns:
            CVaR value
        """
        var = self.historical_var(confidence_level)
        cvar = np.mean(self.portfolio_returns[self.portfolio_returns <= var])
        return cvar
    
    def var_summary(self, confidence_levels=[0.90, 0.95, 0.99]):
        """
        Generate comprehensive VaR summary
        
        Args:
            confidence_levels: list of confidence levels
            
        Returns:
            DataFrame with VaR statistics
        """
        summary = []
        
        for cl in confidence_levels:
            summary.append({
                'Confidence Level': f"{cl*100:.0f}%",
                'Historical VaR': self.historical_var(cl),
                'Parametric VaR': self.parametric_var(cl),
                'CVaR': self.cvar_expected_shortfall(cl)
            })
        
        return pd.DataFrame(summary)

    def kupiec_test_summary(self, confidence_levels=None, alpha=0.05):
        """
        Apply the Kupiec POF backtest to historical and parametric VaR estimates.

        Args:
            confidence_levels: list of confidence levels to test
            alpha: significance level for the hypothesis test

        Returns:
            DataFrame with Kupiec test results for each method and confidence level
        """
        if confidence_levels is None:
            confidence_levels = [0.95, 0.99]

        rows = []
        for cl in confidence_levels:
            historical_var = self.historical_var(cl)
            parametric_var = self.parametric_var(cl)

            for method_name, var_value in [('Historical VaR', historical_var), ('Parametric VaR', parametric_var)]:
                result = kupiec_test(self.portfolio_returns, var_value, cl, alpha=alpha)
                rows.append({
                    'Method': method_name,
                    'Confidence Level': f"{cl*100:.0f}%",
                    'Violations': result['violations'],
                    'Observed Rate': result['observed_rate'],
                    'Expected Rate': result['expected_rate'],
                    'LR Statistic': result['lr_statistic'],
                    'p-value': result['p_value'],
                    'Conclusion': result['conclusion'],
                })

        return pd.DataFrame(rows)


class PortfolioVaRAnalyzer:
    """Analyze VaR for multiple scenarios and compare methods"""
    
    def __init__(self, weights):
        """
        Initialize analyzer
        
        Args:
            weights: portfolio weights
        """
        self.weights = weights
        
    def analyze_scenarios(self, scenarios_dict, confidence_levels=[0.95, 0.99]):
        """
        Analyze VaR across different return scenarios
        
        Args:
            scenarios_dict: dict with scenario names and returns arrays
            confidence_levels: list of confidence levels
            
        Returns:
            dict with VaR results for each scenario
        """
        results = {}
        
        for scenario_name, returns in scenarios_dict.items():
            portfolio_returns = np.dot(returns, self.weights)
            calc = VaRCalculator(portfolio_returns, self.weights)
            
            results[scenario_name] = {
                'mean_return': np.mean(portfolio_returns),
                'std_return': np.std(portfolio_returns),
                'var_summary': calc.var_summary(confidence_levels)
            }
        
        return results
    
    def backtest_var(self, historical_returns, simulated_returns, confidence_level=0.95, window_size=252):
        """
        Backtest VaR by comparing forecasts with actual returns
        
        Args:
            historical_returns: numpy array of shape (n_samples, n_assets)
            simulated_returns: numpy array of shape (n_scenarios, n_assets)
            confidence_level: confidence level
            window_size: rolling window size
            
        Returns:
            backtest results
        """
        portfolio_returns = np.dot(historical_returns, self.weights)
        
        # Calculate VaR on simulated scenarios
        portfolio_simulated = np.dot(simulated_returns, self.weights)
        calculator = VaRCalculator(portfolio_simulated, self.weights)
        var_forecast = calculator.historical_var(confidence_level)
        
        # Count exceedances
        n_observations = len(portfolio_returns)
        n_exceedances = np.sum(portfolio_returns < var_forecast)
        expected_exceedances = n_observations * (1 - confidence_level)
        
        # Kupiec POF test
        exceedance_rate = n_exceedances / n_observations
        expected_rate = 1 - confidence_level
        
        results = {
            'var_forecast': var_forecast,
            'n_exceedances': n_exceedances,
            'expected_exceedances': expected_exceedances,
            'exceedance_rate': exceedance_rate,
            'expected_rate': expected_rate,
            'observations': n_observations
        }
        
        return results
