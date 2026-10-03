import numpy as np
import pandas as pd
from scipy.optimize import minimize
from schemas import Constraints
from math_engine import (
    calculate_portfolio_volatility, 
    calculate_sharpe_ratio, 
    calculate_max_drawdown,
    calculate_annualized_returns,
    calculate_covariance_matrix
)

def run_optimization(strategy: str, returns_matrix: pd.DataFrame, constraints: Constraints) -> np.ndarray:
    num_assets = len(returns_matrix.columns)
    initial_weights = np.array([1.0 / num_assets] * num_assets)
    
    # data needed for objective functions
    cov_matrix = calculate_covariance_matrix(returns_matrix)
    mean_returns = calculate_annualized_returns(returns_matrix)

    
    # Setting CONSTRAINTS 
    # 1: Equality constraint: sum of weights must exactly equal 1.0
    scipy_constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    
    # 2: Bounds: (min_weight, max_weight) for each asset
    bounds = tuple((constraints.min_weight / 100.0, constraints.max_weight / 100.0) for _ in range(num_assets))

    
    # selecting objective function
    if strategy == "equal_weights":
        # our initial weights as it is
        return initial_weights
        
    elif strategy == "minimize_volatility":
        # need to minimize the output of the volatility function
        objective_function = lambda w: calculate_portfolio_volatility(w, cov_matrix)
        
    elif strategy == "maximize_sharpe":
        # scipy only minimizes, to maximize sharpe, we minimize the negative sharpe ratio
        objective_function = lambda w: -calculate_sharpe_ratio(w, mean_returns, cov_matrix)
        
    elif strategy == "minimize_drawdown":
        objective_function = lambda w: calculate_max_drawdown(w, returns_matrix)
        
    # risk parity and factor betas here
    else:
        raise ValueError(f"Unknown optimization strategy: {strategy}")


    # execute scipy slsqp engine
    result = minimize(
        objective_function,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=scipy_constraints,
        options={'maxiter': 1000}
    )
    
    if not result.success:
        raise ValueError("Optimization failed: " + result.message)
        
    return result.x