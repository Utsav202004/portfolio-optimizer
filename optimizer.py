import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional, List
from schemas import Constraints
from math_engine import (
    calculate_portfolio_volatility, 
    calculate_sharpe_ratio, 
    calculate_max_drawdown,
    calculate_risk_parity_loss,
    calculate_annualized_returns,
    calculate_covariance_matrix,
    calculate_asset_factor_betas
)

def run_optimization(
    strategy: str, 
    returns_matrix: pd.DataFrame, 
    constraints: Constraints,
    yields: Optional[np.ndarray] = None,
    factor_matrix: Optional[pd.DataFrame] = None
) -> np.ndarray:
    """
    Executes constrained portfolio optimization using SciPy's SLSQP algorithm.
    
    Supported Strategies:
    - equal_weights: 1 / N allocation baseline
    - risk_parity: Equal risk contribution (ERC)
    - minimize_volatility: Minimum variance portfolio
    - maximize_sharpe_ratio: Maximum Sharpe ratio (tangency portfolio)
    - minimize_drawdown: Minimum historical peak-to-trough drop
    - optimize_factor_exposure: Maximize exposure to Momentum, Value, or Size
    """
    num_assets = len(returns_matrix.columns)
    if num_assets == 0:
        raise ValueError("Cannot optimize an empty portfolio.")

    # 1. Normalize Strategy Name
    clean_strategy = strategy.lower().strip().replace("-", "_").replace(" ", "_")
    if clean_strategy == "maximize_sharpe":
        clean_strategy = "maximize_sharpe_ratio"

    # 2. Extract and Validate Bounds
    min_w = float(constraints.min_weight if constraints.min_weight is not None else 0.0) / 100.0
    max_w = float(constraints.max_weight if constraints.max_weight is not None else 100.0) / 100.0

    if min_w > max_w:
        raise ValueError(
            f"Infeasible constraint: min_weight ({constraints.min_weight}%) "
            f"cannot exceed max_weight ({constraints.max_weight}%)."
        )

    if num_assets * min_w > 1.0 + 1e-6:
        raise ValueError(
            f"Infeasible constraint: {num_assets} assets with minimum weight {constraints.min_weight}% "
            f"require at least {round(num_assets * constraints.min_weight, 2)}%, which exceeds 100%."
        )

    if num_assets * max_w < 1.0 - 1e-6:
        raise ValueError(
            f"Infeasible constraint: {num_assets} assets with maximum weight {constraints.max_weight}% "
            f"can sum to at most {round(num_assets * constraints.max_weight, 2)}%, which cannot reach 100%."
        )

    bounds = tuple((min_w, max_w) for _ in range(num_assets))

    # Fast-path: Equal Weights
    if clean_strategy == "equal_weights":
        equal_weight = 1.0 / num_assets
        if equal_weight < min_w - 1e-6 or equal_weight > max_w + 1e-6:
            raise ValueError(
                f"Equal weights allocation ({round(equal_weight * 100, 2)}% per security) "
                f"violates constraints [{constraints.min_weight}%, {constraints.max_weight}%]."
            )
        return np.array([equal_weight] * num_assets)

    # 3. Covariance Matrix & Expected Returns
    cov_matrix = calculate_covariance_matrix(returns_matrix)
    mean_returns = calculate_annualized_returns(returns_matrix)

    # 4. Formulate Optimization Constraints
    scipy_constraints = [
        # Sum of weights must equal 100% (1.0)
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    ]

    # Dividend Yield Constraint Check
    if constraints.min_dividend_yield is not None:
        if yields is None:
            raise ValueError("Dividend yield data required for dividend yield constraint.")
        target_yield = float(constraints.min_dividend_yield) / 100.0

        # Feasibility check: find theoretical maximum achievable dividend yield
        # Greedily allocate maximum possible weights to highest-yielding assets
        sorted_indices = np.argsort(-yields)
        test_w = np.full(num_assets, min_w)
        remaining = 1.0 - np.sum(test_w)
        for idx in sorted_indices:
            add = min(remaining, max_w - min_w)
            test_w[idx] += add
            remaining -= add
            if remaining <= 1e-8:
                break
        max_achievable_yield = np.sum(test_w * yields)

        if target_yield > max_achievable_yield + 1e-6:
            raise ValueError(
                f"Infeasible constraint: Target dividend yield of {constraints.min_dividend_yield}% "
                f"cannot be achieved. Maximum possible yield under these security limits is "
                f"{round(max_achievable_yield * 100, 2)}%."
            )

        scipy_constraints.append({
            'type': 'ineq',
            'fun': lambda w: np.sum(w * yields) - target_yield
        })

    # CAGR Constraint Check
    if constraints.min_cagr is not None:
        target_cagr = float(constraints.min_cagr) / 100.0
        scipy_constraints.append({
            'type': 'ineq',
            'fun': lambda w: np.sum(w * mean_returns.values) - target_cagr
        })

    # 5. Define Strategy Objective Function
    if clean_strategy == "minimize_volatility":
        objective_function = lambda w: calculate_portfolio_volatility(w, cov_matrix)

    elif clean_strategy == "maximize_sharpe_ratio":
        # SciPy minimizes, so we minimize negative Sharpe ratio
        objective_function = lambda w: -calculate_sharpe_ratio(w, mean_returns, cov_matrix)

    elif clean_strategy == "risk_parity":
        # Equal risk contribution objective
        objective_function = lambda w: calculate_risk_parity_loss(w, cov_matrix.values)

    elif clean_strategy == "minimize_drawdown":
        # Minimize the maximum historical peak-to-trough percentage drop
        objective_function = lambda w: calculate_max_drawdown(w, returns_matrix)

    elif clean_strategy == "optimize_factor_exposure":
        if factor_matrix is None:
            raise ValueError("Factor return data required for factor exposure optimization.")
        
        target_factor = (constraints.target_factor or "momentum").lower().strip()
        asset_betas = calculate_asset_factor_betas(returns_matrix, factor_matrix)
        
        if target_factor not in ["momentum", "value", "size"]:
            raise ValueError(f"Unsupported factor: '{target_factor}'. Supported factors: momentum, value, size")
            
        factor_betas_arr = np.array([asset_betas[t][target_factor] for t in returns_matrix.columns])
        
        # Maximize factor exposure by minimizing the negative factor beta
        objective_function = lambda w: -float(np.sum(w * factor_betas_arr))

    else:
        raise ValueError(
            f"Unknown optimization strategy: '{strategy}'. "
            f"Valid options are: equal_weights, risk_parity, minimize_volatility, "
            f"maximize_sharpe_ratio, minimize_drawdown, optimize_factor_exposure"
        )

    # 6. Initial Guess (Projected into bounds)
    initial_weights = np.clip(np.full(num_assets, 1.0 / num_assets), min_w, max_w)
    initial_weights = initial_weights / np.sum(initial_weights)

    # 7. Execute SLSQP Optimizer
    result = minimize(
        objective_function,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=scipy_constraints,
        options={'maxiter': 1000, 'ftol': 1e-12}
    )

    if not result.success:
        raise ValueError(f"Optimization failed to converge to a feasible solution: {result.message}")

    # Normalize weights so they sum to 1.0 exactly
    opt_weights = np.maximum(result.x, 0.0)
    opt_weights = opt_weights / np.sum(opt_weights)
    return opt_weights