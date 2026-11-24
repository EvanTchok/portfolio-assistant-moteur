"""Portfolio Assistant — the research engine: backtests, honest lab, validation, workshop."""

from .compare import compare, run_with_baseline
from .data import (DataError, Holding, common_window, load_portfolio_csv, load_prices_csv,
                   validate_prices, weights_from_holdings)
from .market_data import fetch_prices, fetch_ticker
from .engine import (BacktestConfig, BacktestError, BacktestResult, DecisionContext,
                     decision_dates, run_backtest)
from .research import Lab, StrategySpec
from . import builder, profiles, universes
from .tracking import Tracker
from .metrics import certainty_equivalent, max_drawdown, periods_per_year, summary
from .strategies import (buy_and_hold, equal_weight, fixed_weights, inverse_volatility,
                         momentum_top_k, RULES)

__all__ = [name for name in dir() if not name.startswith("_")]
