import numpy as np
import pandas as pd

def calculate_annualized_returns(returns_matrix: pd.DataFrame) -> pd.Series:
    # avg daily return, scaled to a year (std no of trading days- 252)
    return returns_matrix.mean() * 252

def calculate_covariance_matrix(returns_matrix: pd.DataFrame) -> pd.DataFrame:
    # matrix showing how every asset correlates with every other asset
    return returns_matrix.cov() * 252

def calculate_portfolio_volatility(weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
    # calculating total portfolio risk : sqrt( weights^T * Covariance_Matrix * weights )
    variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    return np.sqrt(variance)

def calculate_portfolio_return(weights: np.ndarray, mean_returns: pd.Series) -> float:
    # total return of the portfolio based on the weights
    
    return np.sum(mean_returns * weights)

def calculate_sharpe_ratio(weights: np.ndarray, mean_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float = 0.0) -> float:
    # sharpe ratio = (return - risk_free_rate) / volatility
    # higher is better (measures how much return we get per unit of risk)
    p_return = calculate_portfolio_return(weights, mean_returns)
    p_volatility = calculate_portfolio_volatility(weights, cov_matrix)
    
    # handling zero volatility edge case
    if p_volatility == 0:
        return 0.0
        
    return (p_return - risk_free_rate) / p_volatility

def calculate_max_drawdown(weights: np.ndarray, returns_matrix: pd.DataFrame) -> float:
    # calculating the worst possible loss (peak to trough) for this
    
    # daily returns of the combined portfolio
    portfolio_daily_returns = returns_matrix.dot(weights)
    
    # converting daily returns into a cumulative growth curve (starting at 1.0)
    cumulative_returns = (1 + portfolio_daily_returns).cumprod()
    
    # highest peak seen so far
    rolling_max = cumulative_returns.cummax()
    
    # percentage drop from the highest peak to the current value
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    
    # returing the worst drop (most negative number)
    return drawdowns.min()