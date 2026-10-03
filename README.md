# Portfolio Optimizer API

A REST API that replicates the core optimization engine of the Finominal Portfolio Optimizer. Built with FastAPI, Pandas, and SciPy.

## Local Setup Instructions

1. Clone the repository:
`git clone https://github.com/Utsav202004/portfolio-optimizer.git`
`cd portfolio-optimizer`

2. Create a virtual environment:
`python -m venv venv`
`source venv/bin/activate`  # On Windows use: venv\Scripts\activate

3. Install dependencies:
`pip install -r requirements.txt`

4. Run the server:
`uvicorn main:app --reload`

The API will be available at `http://127.0.0.1:8000`. 
Interactive Swagger API documentation is automatically generated at `http://127.0.0.1:8000/docs`.

## Architecture Notes
* **Strict Validation:** Uses Pydantic (`schemas.py`) to enforce constraint boundaries (0-100% limits) and prevent runtime math errors.
* **In-Memory Caching:** `Data.xlsx` is loaded via Pandas once on startup (`data.py`) to prevent disk I/O bottlenecks during POST requests.
* **Math Engine:** SciPy's SLSQP algorithm is utilized for bounded optimization, isolated in `math_engine.py` for testing modularity.

*(Note: Factor Exposure bonus calculation is slated for the next iteration.)*