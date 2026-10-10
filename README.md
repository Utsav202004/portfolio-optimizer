# Portfolio Optimizer API

A quantitative REST API replicating the core optimization engine behind Finominal's Portfolio Optimizer. Built with **FastAPI**, **Pandas**, **NumPy**, and **SciPy (SLSQP)**.

---

##  Features

- **6 Optimization Strategies:**
  1. `equal_weights`: Equal $1/N$ allocation baseline.
  2. `risk_parity`: Equal Risk Contribution (ERC) balancing individual asset volatility contribution.
  3. `minimize_volatility`: Lowest annualized portfolio variance using quadratic covariance form.
  4. `maximize_sharpe_ratio`: Maximum risk-adjusted return tangency portfolio.
  5. `minimize_drawdown`: Minimizes maximum historical peak-to-trough drop.
  6. `optimize_factor_exposure` *(Bonus)*: Maximizes portfolio sensitivity ($\beta$) to equity risk factors (Momentum, Value, Size).
- **Multi-Factor Risk Model (Bonus):** Calculates before-and-after factor betas ($\beta_{\text{val}}, \beta_{\text{mom}}, \beta_{\text{size}}$) via Ordinary Least Squares (OLS) multi-variable regression.
- **Robust Constraint Engine:** Enforces portfolio-level constraints (e.g. Min Dividend Yield, Min CAGR) and asset-level box bounds (`min_weight`, `max_weight`).
- **Strict Infeasibility Protection:** Detects contradictory constraints upfront (e.g. impossible target yields or mutually exclusive bounds) and returns clean HTTP 400 errors instead of producing silent garbage weights.
- **Dynamic Historical Alignment:** Intelligently extracts common date ranges only across the requested securities, preserving maximum historical depth for each portfolio combination.

---

##  Setup & Local Execution

### 1. Prerequisites
- Python 3.10+ installed
- Virtual environment tool

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/Utsav202004/portfolio-optimizer.git
cd portfolio-optimizer

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the API Server
```bash
uvicorn main:app --reload --port 8000
```
- API Endpoint: `http://127.0.0.1:8000/optimize`
- Interactive Swagger / OpenAPI Documentation: `http://127.0.0.1:8000/docs`

### 4. Run Automated Verification Tests
```bash
python test_api.py
```
Executes all 6 official assignment test scenarios plus boundary and infeasibility edge cases.

---

##  Mathematical Formulation & Quant Intuition

### 1. Volatility & Covariance
Portfolio variance is computed as the quadratic form:
$$\sigma_p^2 = w^T \Sigma w$$
where $w$ is the $N \times 1$ weight vector and $\Sigma$ is the $N \times N$ annualized covariance matrix ($252 \times \text{daily\_cov}$). Annualized volatility is $\sigma_p = \sqrt{w^T \Sigma w}$.

### 2. Equal Risk Contribution (Risk Parity)
Rather than equal capital allocation, Risk Parity ensures every asset contributes equally to the total portfolio variance:
$$\text{MCR}_i = (\Sigma w)_i \quad \text{(Marginal Contribution to Risk)}$$
$$\text{RC}_i = w_i \cdot (\Sigma w)_i \quad \text{(Absolute Risk Contribution)}$$
$$\text{RRC}_i = \frac{w_i (\Sigma w)_i}{w^T \Sigma w} = \frac{1}{N} \quad \text{(Relative Risk Contribution)}$$
The objective function minimizes $\sum_{i=1}^N \left(\text{RRC}_i - \frac{1}{N}\right)^2$.

### 3. Factor Betas & Exposure Optimization (Bonus)
Portfolio returns are regressed against Momentum, Value, and Size factor returns over their overlapping history:
$$R_{\text{portfolio}, t} = \alpha + \beta_{\text{mom}} R_{\text{mom}, t} + \beta_{\text{val}} R_{\text{val}, t} + \beta_{\text{size}} R_{\text{size}, t} + \epsilon_t$$
Using Ordinary Least Squares (OLS) via `np.linalg.lstsq`. For Momentum optimization, the engine maximizes $\beta_{\text{mom}} = \sum w_i \beta_{i, \text{mom}}$ subject to weights and constraints.

---

##  API Specification & Example Requests

### Request Schema (`POST /optimize`)
```json
{
  "securities": [
    { "ticker": "IEFA", "weight": 20.0 },
    { "ticker": "GLD", "weight": 20.0 },
    { "ticker": "AGG", "weight": 20.0 },
    { "ticker": "VEA", "weight": 20.0 },
    { "ticker": "SPY", "weight": 20.0 }
  ],
  "strategy": "maximize_sharpe_ratio",
  "constraints": {
    "min_weight": 5.0,
    "max_weight": 40.0,
    "min_dividend_yield": 2.50
  }
}
```

### Response Schema
```json
{
  "optimization_strategy": "maximize_sharpe_ratio",
  "allocation_changes": [
    {
      "ticker": "IEFA",
      "security_name": "iShares Core MSCI EAFE ETF",
      "current_weight": 20.0,
      "optimized_weight": 14.68,
      "change": -5.32
    },
    {
      "ticker": "GLD",
      "security_name": "SPDR Gold Shares",
      "current_weight": 20.0,
      "optimized_weight": 7.14,
      "change": -12.86
    },
    {
      "ticker": "AGG",
      "security_name": "iShares Core US Aggregate Bond ETF",
      "current_weight": 20.0,
      "optimized_weight": 40.0,
      "change": 20.0
    },
    {
      "ticker": "VEA",
      "security_name": "Vanguard Developed Markets Index Fund;ETF",
      "current_weight": 20.0,
      "optimized_weight": 5.0,
      "change": -15.0
    },
    {
      "ticker": "SPY",
      "security_name": "State Street SPDR S&P 500 ETF Trust",
      "current_weight": 20.0,
      "optimized_weight": 33.18,
      "change": 13.18
    }
  ],
  "factor_betas": {
    "current_portfolio": {
      "value": 0.17,
      "momentum": 0.13,
      "size": -0.06
    },
    "optimized_portfolio": {
      "value": 0.12,
      "momentum": 0.1,
      "size": -0.08
    }
  }
}
```

---

##  Verified Test Scenarios

| # | Scenario | Input Portfolio | Strategy | Constraints | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **1** | Basic Equal-Weight | IEFA: 25%, SPY: 75% | `equal_weights` | None | **Passed** |
| **2** | Risk-Based Allocation | VEA: 25%, AGG: 75% | `risk_parity` | None | **Passed** |
| **3** | Lowest-Risk Construction | SPY: 60%, AGG: 30%, GLD: 10% | `minimize_volatility` | None | **Passed** |
| **4** | Risk-Adjusted Return | All 5 Funds (20% each) | `maximize_sharpe_ratio` | None | **Passed** |
| **5** | Complex Constraints | All 5 Funds (20% each) | `maximize_sharpe_ratio` | Min Yield: 2.5%, [5%, 40%] | **Passed** |
| **6** | Momentum Exposure *(Bonus)* | All 5 Funds (20% each) | `optimize_factor_exposure` | Target: Momentum | **Passed** |