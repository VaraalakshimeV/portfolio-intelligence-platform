
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class RiskCalculator:
    """
    Calculate portfolio risk metrics using industry-standard methodologies
    """
    
    def __init__(self, confidence_level: float = 0.95, simulations: int = 10000):
        self.confidence_level = confidence_level
        self.simulations = simulations
        self.trading_days_per_year = 252
    
    def calculate_var(self, returns: np.ndarray, method: str = "historical") -> Dict:
        """
        Calculate Value at Risk (VaR)
        
        Args:
            returns: Array of portfolio returns
            method: 'historical', 'parametric', or 'monte_carlo'
        
        Returns:
            Dict with VaR metrics
        """
        
        if method == "historical":
            var = self._historical_var(returns)
        elif method == "parametric":
            var = self._parametric_var(returns)
        elif method == "monte_carlo":
            var = self._monte_carlo_var(returns)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return var
    
    def _historical_var(self, returns: np.ndarray) -> Dict:
        """Historical VaR - simplest method"""
        sorted_returns = np.sort(returns)
        index = int((1 - self.confidence_level) * len(returns))
        
        var_daily = abs(sorted_returns[index])
        var_monthly = var_daily * np.sqrt(21)  # 21 trading days/month
        
        # Conditional VaR (CVaR) - average of losses beyond VaR
        cvar_daily = abs(np.mean(sorted_returns[:index]))
        
        return {
            'var_95_daily': var_daily,
            'var_95_monthly': var_monthly,
            'cvar_95': cvar_daily,
            'method': 'historical'
        }
    
    def _parametric_var(self, returns: np.ndarray) -> Dict:
        """Parametric VaR - assumes normal distribution"""
        mean = np.mean(returns)
        std = np.std(returns)
        
        # Z-score for 95% confidence
        z_score = stats.norm.ppf(1 - self.confidence_level)
        
        var_daily = abs(mean + z_score * std)
        var_monthly = var_daily * np.sqrt(21)
        
        # CVaR for normal distribution
        cvar_daily = abs(mean - std * stats.norm.pdf(z_score) / (1 - self.confidence_level))
        
        return {
            'var_95_daily': var_daily,
            'var_95_monthly': var_monthly,
            'cvar_95': cvar_daily,
            'method': 'parametric'
        }
    
    def _monte_carlo_var(self, returns: np.ndarray) -> Dict:
        """
        Monte Carlo VaR - most sophisticated method
        Simulates future portfolio paths
        """
        mean = np.mean(returns)
        std = np.std(returns)
        
        # Run simulations
        simulated_returns = np.random.normal(mean, std, self.simulations)
        sorted_returns = np.sort(simulated_returns)
        
        index = int((1 - self.confidence_level) * self.simulations)
        
        var_daily = abs(sorted_returns[index])
        var_monthly = var_daily * np.sqrt(21)
        
        # CVaR
        cvar_daily = abs(np.mean(sorted_returns[:index]))
        
        # Additional Monte Carlo statistics
        var_99 = abs(sorted_returns[int(0.01 * self.simulations)])
        worst_case = abs(sorted_returns[0])
        best_case = sorted_returns[-1]
        
        return {
            'var_95_daily': var_daily,
            'var_95_monthly': var_monthly,
            'var_99_daily': var_99,
            'cvar_95': cvar_daily,
            'worst_case': worst_case,
            'best_case': best_case,
            'method': 'monte_carlo',
            'simulations': self.simulations
        }
    
    def calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.045) -> float:
        """
        Calculate Sharpe Ratio (risk-adjusted returns)
        
        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate (default: 4.5% T-bill)
        
        Returns:
            Sharpe ratio
        """
        mean_return = np.mean(returns) * self.trading_days_per_year
        std_return = np.std(returns) * np.sqrt(self.trading_days_per_year)
        
        sharpe = (mean_return - risk_free_rate) / std_return
        return sharpe
    
    def calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.045) -> float:
        """
        Calculate Sortino Ratio (like Sharpe but only penalizes downside volatility)
        """
        mean_return = np.mean(returns) * self.trading_days_per_year
        
        # Downside deviation (only negative returns)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(self.trading_days_per_year)
        
        if downside_std == 0:
            return np.inf
        
        sortino = (mean_return - risk_free_rate) / downside_std
        return sortino
    
    def calculate_max_drawdown(self, prices: np.ndarray) -> Dict:
        """
        Calculate Maximum Drawdown (worst peak-to-trough decline)
        
        Args:
            prices: Array of portfolio values over time
        
        Returns:
            Dict with max drawdown info
        """
        cumulative = np.maximum.accumulate(prices)
        drawdown = (prices - cumulative) / cumulative
        
        max_dd = np.min(drawdown)
        max_dd_index = np.argmin(drawdown)
        
        # Find the peak before the max drawdown
        peak_index = np.argmax(prices[:max_dd_index]) if max_dd_index > 0 else 0
        
        return {
            'max_drawdown': abs(max_dd),
            'max_drawdown_pct': abs(max_dd) * 100,
            'peak_index': peak_index,
            'trough_index': max_dd_index,
            'recovery_days': len(prices) - max_dd_index if max_dd_index < len(prices) else None
        }
    
    def calculate_beta(self, portfolio_returns: np.ndarray, market_returns: np.ndarray) -> float:
        """
        Calculate Beta (correlation with market)
        Beta > 1: More volatile than market
        Beta < 1: Less volatile than market
        """
        covariance = np.cov(portfolio_returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        
        beta = covariance / market_variance
        return beta
    
    def calculate_alpha(self, portfolio_returns: np.ndarray, market_returns: np.ndarray, 
                       risk_free_rate: float = 0.045) -> float:
        """
        Calculate Alpha (excess return vs. market)
        Positive alpha = outperforming market
        """
        beta = self.calculate_beta(portfolio_returns, market_returns)
        
        portfolio_return = np.mean(portfolio_returns) * self.trading_days_per_year
        market_return = np.mean(market_returns) * self.trading_days_per_year
        
        expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
        alpha = portfolio_return - expected_return
        
        return alpha
    
    def calculate_volatility(self, returns: np.ndarray) -> float:
        """Calculate annualized volatility"""
        return np.std(returns) * np.sqrt(self.trading_days_per_year)
    
    def calculate_correlation_matrix(self, returns_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Calculate correlation matrix between assets
        
        Args:
            returns_dict: Dict of {ticker: returns_array}
        
        Returns:
            Correlation matrix as DataFrame
        """
        df = pd.DataFrame(returns_dict)
        correlation = df.corr()
        return correlation
    
    def calculate_diversification_ratio(self, weights: np.ndarray, 
                                       volatilities: np.ndarray, 
                                       correlation_matrix: np.ndarray) -> float:
        """
        Calculate diversification ratio
        Higher = better diversified
        
        Formula: (Weighted avg of individual volatilities) / (Portfolio volatility)
        """
        # Weighted average of individual volatilities
        weighted_vol = np.sum(weights * volatilities)
        
        # Portfolio volatility
        portfolio_var = np.dot(weights, np.dot(correlation_matrix * np.outer(volatilities, volatilities), weights))
        portfolio_vol = np.sqrt(portfolio_var)
        
        diversification = weighted_vol / portfolio_vol
        return diversification
    
    def stress_test(self, portfolio_value: float, returns: np.ndarray) -> Dict:
        """
        Run stress test scenarios
        
        Returns:
            Impact of different scenarios
        """
        mean = np.mean(returns)
        std = np.std(returns)
        
        scenarios = {
            'market_crash_20pct': portfolio_value * -0.20,
            'market_crash_30pct': portfolio_value * -0.30,
            'black_swan_3sigma': portfolio_value * (mean - 3 * std),
            'flash_crash_10pct_1day': portfolio_value * -0.10,
            'slow_decline_15pct_3months': portfolio_value * -0.15
        }
        
        return scenarios
    
    def calculate_comprehensive_risk(self, returns: np.ndarray, 
                                    portfolio_value: float,
                                    market_returns: np.ndarray = None) -> Dict:
        """
        Calculate all risk metrics in one go
        
        Args:
            returns: Portfolio returns
            portfolio_value: Current portfolio value
            market_returns: Market benchmark returns (optional)
        
        Returns:
            Comprehensive risk metrics
        """
        logger.info("Calculating comprehensive risk metrics...")
        
        # VaR metrics
        var_metrics = self.calculate_var(returns, method="monte_carlo")
        
        # Performance metrics
        sharpe = self.calculate_sharpe_ratio(returns)
        sortino = self.calculate_sortino_ratio(returns)
        volatility = self.calculate_volatility(returns)
        
        # Drawdown
        cumulative_returns = np.cumprod(1 + returns)
        prices = portfolio_value * cumulative_returns
        drawdown = self.calculate_max_drawdown(prices)
        
        # Stress tests
        stress_results = self.stress_test(portfolio_value, returns)
        
        result = {
            **var_metrics,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'volatility': volatility,
            'max_drawdown': drawdown['max_drawdown'],
            'max_drawdown_pct': drawdown['max_drawdown_pct'],
            'stress_tests': stress_results,
            'portfolio_value': portfolio_value
        }
        
        # Add market-relative metrics if market data provided
        if market_returns is not None:
            result['beta'] = self.calculate_beta(returns, market_returns)
            result['alpha'] = self.calculate_alpha(returns, market_returns)
        
        logger.info(f"Risk calculation complete. VaR 95%: ${var_metrics['var_95_daily'] * portfolio_value:,.2f}")
        
        return result


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252)  # 1 year of daily returns
    portfolio_value = 100000
    
    calculator = RiskCalculator()
    risk_metrics = calculator.calculate_comprehensive_risk(returns, portfolio_value)
    
    print("\n=== PORTFOLIO RISK ANALYSIS ===")
    print(f"Portfolio Value: ${portfolio_value:,.2f}")
    print(f"\nValue at Risk (95% confidence):")
    print(f"  Daily VaR: ${risk_metrics['var_95_daily'] * portfolio_value:,.2f}")
    print(f"  Monthly VaR: ${risk_metrics['var_95_monthly'] * portfolio_value:,.2f}")
    print(f"  CVaR (Expected Shortfall): ${risk_metrics['cvar_95'] * portfolio_value:,.2f}")
    print(f"\nPerformance Metrics:")
    print(f"  Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio: {risk_metrics['sortino_ratio']:.2f}")
    print(f"  Volatility (annualized): {risk_metrics['volatility']*100:.2f}%")
    print(f"  Max Drawdown: {risk_metrics['max_drawdown_pct']:.2f}%")
    print(f"\nStress Test Results:")
    for scenario, loss in risk_metrics['stress_tests'].items():
        print(f"  {scenario}: ${loss:,.2f}")