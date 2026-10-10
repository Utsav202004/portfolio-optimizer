from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

# --- Input Schemas ---

class SecurityInput(BaseModel):
    ticker: str = Field(..., description="Ticker symbol of the security, e.g. 'SPY', 'AGG'")
    weight: float = Field(..., ge=0.0, le=100.0, description="Current portfolio weight percentage [0, 100]")

class Constraints(BaseModel):
    min_weight: Optional[float] = Field(0.0, ge=0.0, le=100.0, description="Minimum weight constraint per security (%)")
    max_weight: Optional[float] = Field(100.0, ge=0.0, le=100.0, description="Maximum weight constraint per security (%)")
    min_dividend_yield: Optional[float] = Field(None, ge=0.0, description="Minimum annualized dividend yield (%)")
    min_cagr: Optional[float] = Field(None, description="Minimum annualized return / CAGR (%)")
    target_factor: Optional[str] = Field("momentum", description="Target factor to optimize: 'momentum', 'value', or 'size'")

class OptimizationRequest(BaseModel):
    securities: List[SecurityInput] = Field(..., min_length=1, description="List of securities in the portfolio")
    strategy: str = Field(..., description="Optimization strategy name")
    constraints: Optional[Constraints] = Field(default_factory=Constraints, description="Optimization constraints")

    @field_validator('securities')
    def check_unique_tickers(cls, v):
        tickers = [sec.ticker.upper().strip() for sec in v]
        if len(tickers) != len(set(tickers)):
            raise ValueError("Duplicate tickers are not permitted in a single portfolio.")
        return v

# --- Output Schemas ---

class AllocationChange(BaseModel):
    ticker: str
    security_name: str
    current_weight: float
    optimized_weight: float
    change: float

class FactorBetas(BaseModel):
    value: float
    momentum: float
    size: float

class BetaComparison(BaseModel):
    current_portfolio: FactorBetas
    optimized_portfolio: FactorBetas

class OptimizationResponse(BaseModel):
    optimization_strategy: str
    allocation_changes: List[AllocationChange]
    factor_betas: Optional[BetaComparison] = None