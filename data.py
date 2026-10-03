import pandas as pd

class PortfolioData:
    # Loadinf data once during startup to avoid repeated readings for every web request
    def __init__(self, excel_path: str = "Data.xlsx"):
        # reading all three sheets
        self.fund_info = pd.read_excel(excel_path, sheet_name="Fund Info")
        self.fund_returns = pd.read_excel(excel_path, sheet_name="Fund Returns")
        self.factor_returns = pd.read_excel(excel_path, sheet_name="Factor Returns")
        
        # raw dates to python date time object
        self.fund_returns['date'] = pd.to_datetime(self.fund_returns['date'])
        # pivoting the table, for calculating covariance and variance
        self.returns_matrix = self.fund_returns.pivot(
            index='date', 
            columns='ticker', 
            values='total_return'
        ).dropna()
        
        # Creating quick lookup dictionary for Security Names
        self.ticker_to_name = dict(zip(self.fund_info['ticker'], self.fund_info['fund_name']))
        self.ticker_to_yield = dict(zip(self.fund_info['ticker'], self.fund_info['dividend_yield']))

# Initialize a global instance to hold data in memory
db = PortfolioData()