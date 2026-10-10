import numpy as np
from fastapi import FastAPI, HTTPException, status
from schemas import (
    OptimizationRequest, 
    OptimizationResponse, 
    AllocationChange, 
    FactorBetas, 
    BetaComparison
)
from data import db
from optimizer import run_optimization
from math_engine import calculate_factor_betas

app = FastAPI(
    title="Portfolio Optimizer API",
    description="High-performance quantitative REST API replicating the Finominal Portfolio Optimizer engine.",
    version="1.0.0"
)

@app.get("/", tags=["Health Check"])
def root():
    return {
        "status": "online",
        "message": "Portfolio Optimizer API is operational. Visit /docs for OpenAPI documentation."
    }

@app.post("/optimize", response_model=OptimizationResponse, tags=["Optimization"])
def optimize_endpoint(request: OptimizationRequest):
    # 1. Validate that all requested tickers exist in the database
    valid_tickers = set(db.ticker_to_name.keys())
    for sec in request.securities:
        ticker_clean = sec.ticker.upper().strip()
        if ticker_clean not in valid_tickers:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail=f"Invalid or unrecognized ticker: '{sec.ticker}'. Supported tickers: {list(valid_tickers)}"
            )

    # 2. Validate current weights sum to approximately 100%
    total_current_weight = sum(sec.weight for sec in request.securities)
    if not (99.0 <= total_current_weight <= 101.0):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Input securities' current weights must sum to 100%. Received sum: {round(total_current_weight, 2)}%"
        )

    # 3. Extract dynamically aligned historical return series
    requested_tickers = [sec.ticker.upper().strip() for sec in request.securities]
    subset_returns = db.get_returns_for_tickers(requested_tickers)
    
    if len(subset_returns) < 30:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Insufficient overlapping historical data for requested tickers: {requested_tickers}."
        )

    # 4. Extract dividend yields aligned to requested order
    yields = np.array([db.ticker_to_yield.get(t, 0.0) for t in requested_tickers])

    # 5. Execute Optimization Engine
    try:
        optimized_weights = run_optimization(
            strategy=request.strategy,
            returns_matrix=subset_returns,
            constraints=request.constraints,
            yields=yields,
            factor_matrix=db.factor_matrix
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    # 6. Format Allocation Changes Response
    allocation_changes = []
    opt_weights_pct = np.round(optimized_weights * 100.0, 2)
    
    # Adjust last asset slightly if rounding error makes sum != 100.00
    sum_diff = round(100.0 - float(np.sum(opt_weights_pct)), 2)
    if abs(sum_diff) > 0:
        opt_weights_pct[-1] = round(opt_weights_pct[-1] + sum_diff, 2)

    for i, sec in enumerate(request.securities):
        ticker_clean = sec.ticker.upper().strip()
        opt_w = float(opt_weights_pct[i])
        curr_w = float(sec.weight)
        
        allocation_changes.append(
            AllocationChange(
                ticker=ticker_clean,
                security_name=db.ticker_to_name[ticker_clean],
                current_weight=curr_w,
                optimized_weight=opt_w,
                change=round(opt_w - curr_w, 2)
            )
        )

    # 7. Compute Factor Betas (Bonus Feature)
    current_weights_dec = np.array([sec.weight / 100.0 for sec in request.securities])
    # Normalize current weights to exactly 1.0 for regression
    current_weights_dec = current_weights_dec / np.sum(current_weights_dec)

    curr_factor_betas = calculate_factor_betas(
        weights=current_weights_dec,
        returns_matrix=subset_returns,
        factor_matrix=db.factor_matrix
    )

    opt_factor_betas = calculate_factor_betas(
        weights=optimized_weights,
        returns_matrix=subset_returns,
        factor_matrix=db.factor_matrix
    )

    factor_betas_comparison = BetaComparison(
        current_portfolio=FactorBetas(**curr_factor_betas),
        optimized_portfolio=FactorBetas(**opt_factor_betas)
    )

    return OptimizationResponse(
        optimization_strategy=request.strategy,
        allocation_changes=allocation_changes,
        factor_betas=factor_betas_comparison
    )