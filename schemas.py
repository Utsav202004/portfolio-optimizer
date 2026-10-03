# using pydantic to automatically reject bad requests
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

# Input Schema

class SecurityInput(BaseModel):
    ticker: str
    # Field(...) makes it required, ge and le ensures the weight (keeping the non negative in mind)
    weight: float = Field(..., ge=0, le=100) # Must be between 0 and 100%

class Constraints(BaseModel):
    # Optinal rules that a user can apply to the optimisation engine
    min_weight: Optional[float] = 0.0 # Setting default to 0
    max_weight: Optional[float] = 100.0 # Default to 100
    min_dividend_yield: Optional[float] = None

class OptimizationRequest(BaseModel):
    # The incoming JSON post request
    securities: List[SecurityInput]
    strategy: str 
    constraints: Optional[Constraints] = Constraints()

# Output Schema

class AllocationChange(BaseModel):
    # A single row in the required 'Allocation Changes' response array
    ticker: str
    security_name: str
    current_weight: float
    optimized_weight: float
    change: float

class FactorBetas(BaseModel):
    # Holds the calculate beta values for the three market forces
    value: float
    momentum: float
    size: float

class BetaComparison(BaseModel):
    # Groups the before and after beta calculations
    current_portfolio: FactorBetas
    optimized_portfolio: FactorBetas

class OptimizationResponse(BaseModel):
    # Ootgoing JSON response
    optimization_strategy: str
    allocation_changes: List[AllocationChange]
    factor_betas: Optional[BetaComparison] = None