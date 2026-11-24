"""Price panels and client portfolios: loading and validation.

Rule borrowed from experience: a hole in the data is refused, never forward-filled
silently. A filled gap invents a price, and a backtest on an invented price looks
exactly like a backtest on a real one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


class DataError(ValueError):
    """The input data cannot be used as is; the message says why and where."""


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Checks a wide close-price panel (index = dates, one column per ticker).

    Returns a float64 copy sorted by date. Raises DataError on:
    non-datetime or duplicated index, non-positive prices, holes inside the panel.
    """
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise DataError("prices must be a non-empty DataFrame (dates x tickers)")
    try:
        index = pd.DatetimeIndex(prices.index)
    except (TypeError, ValueError) as exc:
        raise DataError("the price index must be made of dates") from exc
    if index.has_duplicates:
        dupes = index[index.duplicated()].unique()[:5]
        raise DataError(f"duplicated dates in the price index: {list(dupes)}")
    out = prices.copy()
    out.index = index
    out = out.sort_index().astype(np.float64)
    out.columns = [str(c).strip().upper() for c in out.columns]
    if out.columns.duplicated().any():
        raise DataError("duplicated tickers in the price panel")

    values = out.to_numpy()
    if np.any(np.isfinite(values) & (values <= 0.0)):
        bad = out.columns[(np.isfinite(values) & (values <= 0.0)).any(axis=0)].tolist()
        raise DataError(f"non-positive prices for {bad}")

    holes = {}
    for ticker in out.columns:
        col = out[ticker]
        first, last = col.first_valid_index(), col.last_valid_index()
        if first is None:
            raise DataError(f"{ticker} has no price at all")
        inside = col.loc[first:last]
        missing = inside.index[inside.isna()]
        if len(missing):
            holes[ticker] = (len(missing), missing[0])
    if holes:
        detail = ", ".join(f"{t}: {n} missing from {d.date()}" for t, (n, d) in holes.items())
        raise DataError(
            f"holes inside the price history ({detail}). They are refused, not "
            "forward-filled: clean the source or drop the ticker."
        )
    return out


def common_window(prices: pd.DataFrame) -> pd.DataFrame:
    """The date range where every ticker quotes. Reports what was cut."""
    starts = prices.apply(lambda c: c.first_valid_index())
    ends = prices.apply(lambda c: c.last_valid_index())
    start, end = starts.max(), ends.min()
    if start >= end:
        raise DataError(
            f"no common window: latest start {start.date()} ({starts.idxmax()}) is not "
            f"before earliest end {end.date()} ({ends.idxmin()})"
        )
    return prices.loc[start:end]


def load_prices_csv(path: str | Path, date_column: str = "date") -> pd.DataFrame:
    """Wide CSV: one date column, one column per ticker."""
    frame = pd.read_csv(path)
    lookup = {c.lower(): c for c in frame.columns}
    if date_column.lower() not in lookup:
        raise DataError(f"no '{date_column}' column in {path}")
    frame = frame.set_index(lookup[date_column.lower()])
    frame.index = pd.to_datetime(frame.index)
    return validate_prices(frame)


@dataclass(slots=True)
class Holding:
    ticker: str
    quantity: float
    currency: str = "EUR"


def load_portfolio_csv(path: str | Path) -> list[Holding]:
    """Client portfolio export: columns ticker, quantity (and optionally currency).

    Column names are matched case-insensitively and in French or English.
    """
    frame = pd.read_csv(path)
    aliases = {
        "ticker": {"ticker", "symbole", "code", "isin"},
        "quantity": {"quantity", "quantite", "quantité", "qty", "nombre"},
        "currency": {"currency", "devise"},
    }
    mapping = {}
    for target, names in aliases.items():
        for column in frame.columns:
            if str(column).strip().lower() in names:
                mapping[target] = column
                break
    for required in ("ticker", "quantity"):
        if required not in mapping:
            raise DataError(f"portfolio file needs a '{required}' column, found {list(frame.columns)}")
    holdings = []
    for row_number, row in frame.iterrows():
        quantity = float(row[mapping["quantity"]])
        if not np.isfinite(quantity) or quantity < 0:
            raise DataError(f"row {row_number + 2}: invalid quantity {row[mapping['quantity']]!r}")
        currency = str(row[mapping["currency"]]).strip().upper() if "currency" in mapping else "EUR"
        holdings.append(Holding(str(row[mapping["ticker"]]).strip().upper(), quantity, currency))
    return holdings


def weights_from_holdings(holdings: list[Holding], prices: pd.DataFrame, at=None) -> dict[str, float]:
    """Current weights of a client portfolio, valued at the last date <= ``at``."""
    priced = prices.loc[:at] if at is not None else prices
    if priced.empty:
        raise DataError("no price on or before the valuation date")
    last = priced.iloc[-1]
    foreign = sorted({h.currency for h in holdings} - {"EUR"})
    if foreign:
        raise DataError(f"currencies {foreign} need FX conversion, not supported yet in the MVP")
    values = {}
    for holding in holdings:
        if holding.ticker not in last.index or not np.isfinite(last[holding.ticker]):
            raise DataError(f"no price for {holding.ticker} at the valuation date")
        values[holding.ticker] = values.get(holding.ticker, 0.0) + holding.quantity * float(last[holding.ticker])
    total = sum(values.values())
    if total <= 0:
        raise DataError("the portfolio is worth nothing at the valuation date")
    return {t: v / total for t, v in values.items()}
