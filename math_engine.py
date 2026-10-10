import numpy as np
import pandas as pd
from typing import Dict

def calculate_annualized_returns(returns_matrix: pd.DataFrame) -> pd.Series:
    """
    Computes annualized arithmetic mean returns assuming 252 trading days per year.
    Formula: daily_mean * 252
    """
    return returns_matrix.mean() * 252

def calculate_covariance_matrix(returns_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Annualized covariance matrix showing how each asset co-moves with all other assets.
    Formula: daily_cov * 252
    """
    return returns_matrix.cov() * 252

def calculate_portfolio_volatility(weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
    """
    Total portfolio risk (annualized standard deviation).
    Formula: sqrt( weights^T * Covariance_Matrix * weights )
    """
    variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    return float(np.sqrt(np.maximum(variance, 0.0)))

def calculate_portfolio_return(weights: np.ndarray, mean_returns: pd.Series) -> float:
    """
    Total annualized expected portfolio return.
    Formula: sum( weights * mean_returns )
    """
    return float(np.sum(weights * mean_returns))

def calculate_sharpe_ratio(weights: np.ndarray, mean_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float = 0.0) -> float:
    """
    Sharpe Ratio: Measures risk-adjusted excess return per unit of volatility.
    Formula: (portfolio_return - risk_free_rate) / portfolio_volatility
    """
    p_return = calculate_portfolio_return(weights, mean_returns)
    p_volatility = calculate_portfolio_volatility(weights, cov_matrix)
    
    if p_volatility <= 1e-8:
        return 0.0
        
    return float((p_return - risk_free_rate) / p_volatility)

def calculate_max_drawdown(weights: np.ndarray, returns_matrix: pd.DataFrame) -> float:
    """
    Computes the maximum peak-to-trough drop over the historical time series.
    Returns a positive percentage drop (e.g., 0.15 for 15% drawdown) so that minimizing this value minimizes the worst historical drop.
    """
    portfolio_daily_returns = returns_matrix.dot(weights)
    cumulative_returns = (1.0 + portfolio_daily_returns).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    # drawdowns are negative or zero (e.g. -0.20 for 20% drop). We return the magnitude: -min(drawdowns) >= 0.
    return float(-drawdowns.min())

def calculate_risk_parity_loss(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    Equal Risk Contribution (ERC) / Risk Parity objective function.
    
    The marginal contribution to risk (MCR) of asset i is: (cov * w)_i
    The absolute risk contribution (RC_i) is: w_i * MCR_i
    The relative risk contribution (RRC_i) is: RC_i / total_portfolio_variance
    
    In Risk Parity, every asset must contribute equally: RRC_i = 1 / N.
    We minimize the sum of squared differences: sum_i (RRC_i - 1/N)^2.
    """
    n = len(weights)
    port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
    if port_var <= 1e-12:
        return 0.0
    
    marginal_risk_contrib = np.dot(cov_matrix, weights)
    risk_contrib = weights * marginal_risk_contrib
    relative_risk_contrib = risk_contrib / port_var
    
    target_contrib = 1.0 / n
    return float(np.sum((relative_risk_contrib - target_contrib) ** 2))

def calculate_factor_betas(weights: np.ndarray,returns_matrix: pd.DataFrame,factor_matrix: pd.DataFrame) -> Dict[str, float]:
    """
    Regresses the portfolio historical daily returns against the three market factor series:
    Momentum, Value, and Size across their common historical date range.
    
    Model:
    R_portfolio,t = alpha + beta_1(Momentum) + beta_2(Value) + beta_3(Size) + error
    
    Uses Ordinary Least Squares (OLS) via np.linalg.lstsq.
    """
    # 1. Construct portfolio daily return series
    portfolio_daily_returns = returns_matrix.dot(weights)
    
    # 2. Align on common dates with factor returns
    joined = pd.DataFrame({'portfolio': portfolio_daily_returns}).join(factor_matrix).dropna()
    
    if len(joined) == 0:
        return {'value': 0.0, 'momentum': 0.0, 'size': 0.0}
        
    y = joined['portfolio'].values
    # Order: Momentum Factor, Value Factor, Size Factor
    x_factors = joined[['Momentum Factor', 'Value Factor', 'Size Factor']].values
    
    # Add constant column (intercept / alpha)
    X = np.column_stack([np.ones(len(y)), x_factors])
    
    # Solve OLS: beta = (X^T * X)^(-1) * X^T * y
    betas, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta_mom, beta_val, beta_size = betas
    
    return {
        'value': round(float(beta_val), 2),
        'momentum': round(float(beta_mom), 2),
        'size': round(float(beta_size), 2)
    }

def calculate_asset_factor_betas(returns_matrix: pd.DataFrame,factor_matrix: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Computes individual factor betas for each asset against Momentum, Value, and Size.
    Useful for direct linear exposure optimization.
    """
    joined = returns_matrix.join(factor_matrix).dropna()
    x_factors = joined[['Momentum Factor', 'Value Factor', 'Size Factor']].values
    X = np.column_stack([np.ones(len(joined)), x_factors])
    
    asset_betas = {}
    for ticker in returns_matrix.columns:
        y = joined[ticker].values
        betas, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        asset_betas[ticker] = {
            'alpha': float(betas[0]),
            'momentum': float(betas[1]),
            'value': float(betas[2]),
            'size': float(betas[3])
        }
    return asset_betas