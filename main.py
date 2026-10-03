from fastapi import FastAPI, HTTPException
from schemas import OptimizationRequest, OptimizationResponse, AllocationChange
from data import db
from optimizer import run_optimization

app = FastAPI(title="Portfolio Optimizer API")

@app.post("/optimize", response_model=OptimizationResponse)
def optimize_endpoint(request: OptimizationRequest):
    # 1. checking all requested tickers  exist in our excel data
    valid_tickers = db.returns_matrix.columns.tolist()
    for sec in request.securities:
        if sec.ticker not in valid_tickers:
            raise HTTPException(status_code=400, detail=f"Invalid ticker: {sec.ticker}")
            
    # 2. reordering the data matrix to match the order the user sent the tickers
    requested_tickers = [sec.ticker for sec in request.securities]
    subset_returns = db.returns_matrix[requested_tickers]
    
    # 3. executing the core optimization engine
    try:
        optimized_weights = run_optimization(
            strategy=request.strategy,
            returns_matrix=subset_returns,
            constraints=request.constraints
        )
    except ValueError as e:
        # incase scipy fails (like impossible constraints), returning 400 error
        raise HTTPException(status_code=400, detail=str(e))
        
    # 4. formatting the final JSON response to match the exact requirements
    allocation_changes = []
    
    for i, sec in enumerate(request.securities):
        # converting internal decimal weight back to percentage (0.25 to 25.0)
        opt_w_pct = round(optimized_weights[i] * 100, 2)
        current_w_pct = sec.weight
        
        allocation_changes.append(
            AllocationChange(
                ticker=sec.ticker,
                security_name=db.ticker_to_name[sec.ticker],
                current_weight=current_w_pct,
                optimized_weight=opt_w_pct,
                change=round(opt_w_pct - current_w_pct, 2)
            )
        )
        
    return OptimizationResponse(
        optimization_strategy=request.strategy,
        allocation_changes=allocation_changes
    )