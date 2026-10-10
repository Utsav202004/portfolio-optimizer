import pandas as pd
from typing import List

class PortfolioData:
    """
    Loads Excel data once during startup to avoid repeated disk I/O bottlenecks.
    Preserves raw return histories so we can dynamically compute the longest available overlapping history for any subset of requested securities.
    """
    def __init__(self, excel_path: str = "Data.xlsx"):
        # Load sheets from Excel
        self.fund_info = pd.read_excel(excel_path, sheet_name="Fund Info")
        self.fund_returns = pd.read_excel(excel_path, sheet_name="Fund Returns")
        self.factor_returns = pd.read_excel(excel_path, sheet_name="Factor Returns")
        
        # Standardize dates
        self.fund_returns['date'] = pd.to_datetime(self.fund_returns['date'])
        self.factor_returns['date'] = pd.to_datetime(self.factor_returns['date'])
        
        # Full pivoted returns matrices (without premature global dropna)
        self.all_fund_returns = self.fund_returns.pivot(
            index='date', 
            columns='ticker', 
            values='total_return'
        ).sort_index()

        self.factor_matrix = self.factor_returns.pivot(
            index='date', 
            columns='index_ticker', 
            values='total_return'
        ).sort_index()
        
        # Quick lookup dictionaries
        self.ticker_to_name = dict(zip(self.fund_info['ticker'], self.fund_info['fund_name']))
        self.ticker_to_yield = dict(zip(self.fund_info['ticker'], self.fund_info['dividend_yield'].fillna(0.0)))

    def get_returns_for_tickers(self, tickers: List[str]) -> pd.DataFrame:
        """
        Dynamically extracts and drops NaNs only for the requested subset of tickers, maximizing the usable historical date window for those specific funds.
        """
        return self.all_fund_returns[tickers].dropna()

# Global database instance held in memory
db = PortfolioData()