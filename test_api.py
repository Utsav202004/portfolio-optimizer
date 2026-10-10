"""
Automated Test Suite, Validates all 6 required evaluation test scenarios and edge cases.
Can be executed via standard python: `python test_api.py`
"""

import sys
from main import optimize_endpoint
from schemas import OptimizationRequest, SecurityInput, Constraints
from fastapi import HTTPException

def test_case_1_equal_weights():
    req = OptimizationRequest(
        securities=[
            SecurityInput(ticker='IEFA', weight=25.0),
            SecurityInput(ticker='SPY', weight=75.0)
        ],
        strategy='equal_weights'
    )
    res = optimize_endpoint(req)
    assert res.optimization_strategy == 'equal_weights'
    weights = {c.ticker: c.optimized_weight for c in res.allocation_changes}
    assert weights['IEFA'] == 50.0
    assert weights['SPY'] == 50.0
    assert round(sum(weights.values()), 2) == 100.0
    print("  [✓] Case 1: Equal Weights PASSED")

def test_case_2_risk_parity():
    req = OptimizationRequest(
        securities=[
            SecurityInput(ticker='VEA', weight=25.0),
            SecurityInput(ticker='AGG', weight=75.0)
        ],
        strategy='risk_parity'
    )
    res = optimize_endpoint(req)
    assert res.optimization_strategy == 'risk_parity'
    weights = {c.ticker: c.optimized_weight for c in res.allocation_changes}
    assert round(sum(weights.values()), 2) == 100.0
    # AGG is much less volatile than VEA, so AGG must receive significantly more weight
    assert weights['AGG'] > weights['VEA']
    assert 75.0 <= weights['AGG'] <= 85.0
    print("  [✓] Case 2: Risk Parity PASSED")

def test_case_3_minimize_volatility():
    req = OptimizationRequest(
        securities=[
            SecurityInput(ticker='SPY', weight=60.0),
            SecurityInput(ticker='AGG', weight=30.0),
            SecurityInput(ticker='GLD', weight=10.0)
        ],
        strategy='minimize_volatility'
    )
    res = optimize_endpoint(req)
    weights = {c.ticker: c.optimized_weight for c in res.allocation_changes}
    assert round(sum(weights.values()), 2) == 100.0
    # Low-risk portfolio is heavily anchored in aggregate bonds
    assert weights['AGG'] > 85.0
    print("  [✓] Case 3: Minimize Volatility PASSED")

def test_case_4_maximize_sharpe_unconstrained():
    req = OptimizationRequest(
        securities=[
            SecurityInput(ticker='IEFA', weight=20.0),
            SecurityInput(ticker='GLD', weight=20.0),
            SecurityInput(ticker='AGG', weight=20.0),
            SecurityInput(ticker='VEA', weight=20.0),
            SecurityInput(ticker='SPY', weight=20.0)
        ],
        strategy='maximize_sharpe_ratio'
    )
    res = optimize_endpoint(req)
    weights = {c.ticker: c.optimized_weight for c in res.allocation_changes}
    assert round(sum(weights.values()), 2) == 100.0
    assert all(w >= 0.0 for w in weights.values())
    print("  [✓] Case 4: Maximize Sharpe (Unconstrained) PASSED")

def test_case_5_maximize_sharpe_constrained():
    req = OptimizationRequest(
        securities=[
            SecurityInput(ticker='IEFA', weight=20.0),
            SecurityInput(ticker='GLD', weight=20.0),
            SecurityInput(ticker='AGG', weight=20.0),
            SecurityInput(ticker='VEA', weight=20.0),
            SecurityInput(ticker='SPY', weight=20.0)
        ],
        strategy='maximize_sharpe_ratio',
        constraints=Constraints(
            min_dividend_yield=2.5,
            min_weight=5.0,
            max_weight=40.0
        )
    )
    res = optimize_endpoint(req)
    weights = {c.ticker: c.optimized_weight for c in res.allocation_changes}
    assert round(sum(weights.values()), 2) == 100.0
    # Every security must respect 5% min and 40% max bounds
    for ticker, w in weights.items():
        assert 4.99 <= w <= 40.01, f"{ticker} weight {w}% violates [5%, 40%]"
    print("  [✓] Case 5: Maximize Sharpe (Constrained) PASSED")

def test_case_6_factor_exposure_momentum():
    req = OptimizationRequest(
        securities=[
            SecurityInput(ticker='IEFA', weight=20.0),
            SecurityInput(ticker='GLD', weight=20.0),
            SecurityInput(ticker='AGG', weight=20.0),
            SecurityInput(ticker='VEA', weight=20.0),
            SecurityInput(ticker='SPY', weight=20.0)
        ],
        strategy='optimize_factor_exposure',
        constraints=Constraints(target_factor='momentum')
    )
    res = optimize_endpoint(req)
    assert res.factor_betas is not None
    curr_mom = res.factor_betas.current_portfolio.momentum
    opt_mom = res.factor_betas.optimized_portfolio.momentum
    assert opt_mom > curr_mom, f"Optimized momentum ({opt_mom}) must be higher than current ({curr_mom})"
    print("  [✓] Case 6: Factor Exposure (Momentum) PASSED")

def test_infeasible_dividend_yield():
    req = OptimizationRequest(
        securities=[
            SecurityInput(ticker='IEFA', weight=20.0),
            SecurityInput(ticker='GLD', weight=20.0),
            SecurityInput(ticker='AGG', weight=20.0),
            SecurityInput(ticker='VEA', weight=20.0),
            SecurityInput(ticker='SPY', weight=20.0)
        ],
        strategy='maximize_sharpe_ratio',
        constraints=Constraints(min_dividend_yield=10.0)
    )
    caught = False
    try:
        optimize_endpoint(req)
    except HTTPException as e:
        caught = True
        assert e.status_code == 400
        assert "Infeasible constraint" in e.detail
    assert caught, "Should have raised HTTPException for impossible dividend yield"
    print("  [✓] Case 7: Infeasible Yield Rejection PASSED")

def test_infeasible_weight_bounds():
    req = OptimizationRequest(
        securities=[
            SecurityInput(ticker='IEFA', weight=20.0),
            SecurityInput(ticker='GLD', weight=20.0),
            SecurityInput(ticker='AGG', weight=20.0),
            SecurityInput(ticker='VEA', weight=20.0),
            SecurityInput(ticker='SPY', weight=20.0)
        ],
        strategy='maximize_sharpe_ratio',
        constraints=Constraints(min_weight=30.0)
    )
    caught = False
    try:
        optimize_endpoint(req)
    except HTTPException as e:
        caught = True
        assert e.status_code == 400
        assert "Infeasible constraint" in e.detail
    assert caught, "Should have raised HTTPException for impossible weight bounds"
    print("  [✓] Case 8: Infeasible Bounds Rejection PASSED")

if __name__ == "__main__":
    print("\nRunning Automated Verification Tests...")
    test_case_1_equal_weights()
    test_case_2_risk_parity()
    test_case_3_minimize_volatility()
    test_case_4_maximize_sharpe_unconstrained()
    test_case_5_maximize_sharpe_constrained()
    test_case_6_factor_exposure_momentum()
    test_infeasible_dividend_yield()
    test_infeasible_weight_bounds()
    print("\n🎉 ALL 8 TESTS PASSED SUCCESSFULLY!\n")
