import datetime
from pathlib import Path
from typing import Iterable, Union
import pandas as pd


class StockMarketAnalyzer:
    def __init__(
        self,
        market_data: dict[str, pd.DataFrame],
        sector: str = "NASDAQ",
        date_col: str = "Date",
        price_col: str = "Open",
    ):
        self._market_data = market_data
        self._sector = sector
        self._date_col = date_col
        self._price_col = price_col

    def compute_sector_ratio(
        self, date: datetime.datetime, days_before: int = 1, days_after: int = 1
    ):
        return self.compute_stock_ratio(self._sector, date, days_before, days_after)

    def query(self, company: str, cond: Union[pd.Series, str]):
        return self._market_data[company][cond]

    def get_stock_price(self, company: str, date: datetime.datetime) -> float:
        stocks = self._market_data[company]
        query = stocks[self._date_col] == date
        return self.query(company, query)[self._price_col].iloc[0]

    def get_previous_price(
        self, company: str, date: datetime.datetime, offset: int = 1
    ):
        stocks = self._market_data[company]
        query = stocks[self._date_col] < date
        value = self.query(company, query)[self._price_col].iloc[-offset]
        return float(value)

    def get_next_price(self, company: str, date: datetime.datetime, offset: int = 1):
        stocks = self._market_data[company]
        query = stocks[self._date_col] > date
        value = self.query(company, query)[self._price_col].iloc[offset - 1]
        return float(value)

    def compute_stock_ratio(
        self,
        company: str,
        date: datetime.datetime,
        days_before: int = 1,
        days_after: int = 1,
    ):
        before_price = self.get_previous_price(company, date, offset=days_before)
        after_price = self.get_next_price(company, date, offset=days_after)
        stock_ratio = after_price / before_price
        return stock_ratio

    def beats_market(
        self,
        company: str,
        date: datetime.datetime,
        days_before: int = 1,
        days_after: int = 1,
    ):
        stock_ratio = self.compute_stock_ratio(company, date, days_before, days_after)
        sector_ratio = self.compute_sector_ratio(date, days_before, days_after)

        return stock_ratio > sector_ratio


def load_stock_prices(files: Iterable[Path]) -> dict[str, pd.DataFrame]:
    """
    Load stock prices from CSV files into a dictionary of DataFrames.

    Args:
        files (Iterable[Path]): An iterable of Path objects representing the
        CSV files to load.

    Returns:
        dict[str, pd.DataFrame]: A dictionary where the keys are the company names
        (extracted from the file names) and the values are pandas DataFrames
        containing the stock prices for each company.
    """

    market_data = {}

    for f in files:
        path = f.as_posix()
        company = f.stem
        market_data[company] = pd.read_csv(
            path, usecols=["Date", "Open"], parse_dates=["Date"]
        )

    return market_data
