"""Periodic-rebalance backtester for long-only portfolios.

Design rules (each one answers a real failure seen in a mature framework):

* The decision function only ever receives prices up to the decision date. It cannot
  look ahead because the future is not in the object it is given.
* Orders are executed ``execution_delay`` bars after the decision (default 1). A
  decision taken at Monday's close trades at Tuesday's close.
* Accounting identity ``nav == cash + invested`` is checked on every bar. A run that
  breaks it raises instead of publishing a number.
* Performance is read on the NAV, never compounded from trade returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import validate_prices

IDENTITY_TOLERANCE = 1e-10


class BacktestError(RuntimeError):
    """The backtest itself is inconsistent; never silenced."""


@dataclass(slots=True)
class DecisionContext:
    """What a weighting rule may see when it decides."""

    date: pd.Timestamp
    history: pd.DataFrame            # prices up to and including ``date``
    current_weights: pd.Series       # weights held at the close of ``date``
    nav: float

    @property
    def returns(self) -> pd.DataFrame:
        return self.history.pct_change().iloc[1:]


WeightFn = Callable[[DecisionContext], Mapping[str, float] | pd.Series | None]


@dataclass(slots=True)
class BacktestConfig:
    initial_capital: float = 100_000.0
    rebalance: str | Sequence = "monthly"   # daily|weekly|monthly|quarterly|yearly|once|<dates>
    execution_delay: int = 1
    lookback: int = 0                        # bars of history required before the first decision
    fee_rate: float = 0.001                  # proportional cost per traded notional (10 bp)
    cash_rate: float = 0.0                   # annual rate earned by cash, act/365
    drift_threshold: float = 0.0             # skip a rebalance if max |w - target| below this
    # prices of a money-market fund: cash then earns what that fund earned between two bars
    # (it replaces cash_rate). Before the fund's first price, cash earns nothing.
    cash_prices: pd.Series | None = None

    def __post_init__(self):
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be > 0")
        if int(self.execution_delay) < 1:
            raise ValueError("execution_delay must be >= 1: same-bar execution is look-ahead")
        if self.fee_rate < 0 or self.fee_rate >= 0.5:
            raise ValueError("fee_rate must be in [0, 0.5)")
        if self.lookback < 0:
            raise ValueError("lookback must be >= 0")


@dataclass(slots=True)
class BacktestResult:
    nav: pd.Series
    cash: pd.Series
    invested: pd.Series
    weights: pd.DataFrame
    units: pd.DataFrame
    trades: pd.DataFrame
    decisions: pd.DataFrame
    config: BacktestConfig
    label: str = "strategy"
    fees_paid: float = 0.0
    notes: list[str] = field(default_factory=list)
    prices: pd.DataFrame | None = None

    @property
    def returns(self) -> pd.Series:
        return self.nav.pct_change().iloc[1:]

    @property
    def n_decisions(self) -> int:
        return int(len(self.decisions))

    @property
    def turnover_annual(self) -> float:
        if self.trades.empty:
            return 0.0
        traded = self.trades.groupby("date")["notional"].apply(lambda s: s.abs().sum())
        nav_at = self.nav.reindex(traded.index)
        years = max((self.nav.index[-1] - self.nav.index[0]).days / 365.25, 1e-9)
        return float((traded / nav_at).sum() / 2.0 / years)


_FREQ = {"weekly": "W", "monthly": "M", "quarterly": "Q", "yearly": "Y"}


def decision_dates(index: pd.DatetimeIndex, rebalance, lookback: int) -> list[int]:
    """Bar positions where a decision is taken: the LAST bar of each period.

    Deciding on the last close of a period and trading on the next bar keeps the
    rebalance inside the new period, and is knowable before the run.
    """
    first = int(lookback)
    if first >= len(index) - 1:
        raise ValueError(f"lookback={lookback} leaves no bar to trade on ({len(index)} bars)")
    if isinstance(rebalance, str):
        key = rebalance.strip().lower()
        if key == "daily":
            return list(range(first, len(index) - 1))
        if key == "once":
            return [first]
        if key not in _FREQ:
            raise ValueError(f"unknown rebalance {rebalance!r}")
        periods = index.to_period(_FREQ[key])
        last_of_period = np.flatnonzero(periods[1:] != periods[:-1])  # bar before a period change
        bars = sorted({first, *[int(b) for b in last_of_period if b >= first]})
        return [b for b in bars if b < len(index) - 1]
    wanted = pd.DatetimeIndex(rebalance)
    positions = index.get_indexer(wanted)
    if (positions < 0).any():
        raise ValueError(f"rebalance dates not in the price index: {list(wanted[positions < 0][:5])}")
    return sorted(int(p) for p in positions if first <= p < len(index) - 1)


def _clean_weights(raw, tickers: pd.Index) -> pd.Series:
    if isinstance(raw, pd.Series):
        keys, values = list(raw.index), raw.to_numpy(dtype=np.float64)
    else:
        items = dict(raw)
        keys, values = list(items), np.array([float(v) for v in items.values()], dtype=np.float64)
    names = [str(t).strip().upper() for t in keys]
    if len(set(names)) != len(names):
        raise ValueError("cannot reindex on an axis with duplicate labels")
    position = {t: i for i, t in enumerate(tickers)}
    unknown = sorted(set(names) - set(position))
    if unknown:
        raise ValueError(f"weights reference unknown tickers {unknown}")
    if not np.isfinite(values).all():
        raise ValueError("weights must be finite")
    if (values < -1e-12).any():
        raise ValueError("long-only MVP: negative weights are refused")
    total = float(values.sum())
    if total > 1.0 + 1e-9:
        raise ValueError(f"weights sum to {total:.6f} > 1: leverage is not supported in the MVP")
    out = np.zeros(len(tickers))
    for name, v in zip(names, values):
        out[position[name]] = v
    return pd.Series(np.maximum(out, 0.0), index=tickers)


def run_backtest(prices: pd.DataFrame, weight_fn: WeightFn, config: BacktestConfig | None = None,
                 label: str = "strategy", *, reference: bool = False) -> BacktestResult:
    """``reference=True`` walks every bar one by one (the reference implementation); by default
    the holding periods between two orders are computed at once, which gives the same numbers
    (tested to 1e-10) several times faster."""
    if not reference:
        return _run_fast(prices, weight_fn, config or BacktestConfig(), label)
    config = config or BacktestConfig()
    prices = validate_prices(prices)
    if prices.isna().any().any():
        raise ValueError("the backtest needs a full panel; use data.common_window() first")
    index, tickers = prices.index, prices.columns
    px = prices.to_numpy()
    n_bars, n_assets = px.shape

    decide_at = decision_dates(index, config.rebalance, config.lookback)
    exec_at: dict[int, list[int]] = {}
    for d in decide_at:
        e = d + int(config.execution_delay)
        if e < n_bars:
            exec_at.setdefault(e, []).append(d)

    cash = np.empty(n_bars)
    invested = np.empty(n_bars)
    units_hist = np.zeros((n_bars, n_assets))
    units = np.zeros(n_assets)
    cash_now = float(config.initial_capital)
    pending: dict[int, pd.Series] = {}
    trades, decisions, notes = [], [], []
    fees_total = 0.0
    fees_bar = np.zeros(n_bars)
    interest_bar = np.zeros(n_bars)
    # calendar days between bars, independent of the datetime resolution (ns/us/s)
    day_gap = np.r_[0.0, np.diff(index.to_numpy(dtype="datetime64[ns]")).astype("int64") / 86_400e9]
    cash_growth = None
    if config.cash_prices is not None:
        level = config.cash_prices.sort_index()
        level = level[~level.index.duplicated(keep="last")].reindex(index, method="ffill")   # past prices only
        cash_growth = (level / level.shift(1)).fillna(1.0).to_numpy()

    for t in range(n_bars):
        # 1. cash interest accrued since the previous bar
        if t > 0 and cash_growth is not None and cash_now > 0:
            accrued = cash_now * (cash_growth[t] - 1.0)          # negative when rates were negative
            cash_now += accrued
            interest_bar[t] = accrued
        elif t > 0 and config.cash_rate and cash_now > 0:
            accrued = cash_now * ((1.0 + config.cash_rate) ** (day_gap[t] / 365.0) - 1.0)
            cash_now += accrued
            interest_bar[t] = accrued

        # 2. execute orders decided earlier (oldest decision wins only if several land here)
        live = [d for d in exec_at.get(t, ()) if d in pending]
        if live:
            # several decisions landing on one bar: the latest one wins
            target = pending.pop(max(live))
            for older in live:
                pending.pop(older, None)
            value_now = units * px[t]
            nav_pre = cash_now + value_now.sum()
            w = target.to_numpy()
            # fees reduce the NAV the targets are applied to: solve N' = N - f*sum|w N' - v|
            nav_post = nav_pre
            for _ in range(20):
                fee = config.fee_rate * np.abs(w * nav_post - value_now).sum()
                nav_post_new = nav_pre - fee
                if abs(nav_post_new - nav_post) < 1e-12 * max(nav_pre, 1.0):
                    nav_post = nav_post_new
                    break
                nav_post = nav_post_new
            new_units = w * nav_post / px[t]
            delta = new_units - units
            notional = delta * px[t]
            fee = config.fee_rate * np.abs(notional).sum()
            cash_now = cash_now - notional.sum() - fee
            fees_total += fee
            units = new_units
            for j in np.flatnonzero(np.abs(notional) > 1e-9):
                trades.append({"date": index[t], "ticker": tickers[j], "units": delta[j],
                               "price": px[t, j], "notional": notional[j],
                               "fee": config.fee_rate * abs(notional[j])})
            fees_bar[t] += fee
            if cash_now < -1e-6 * max(nav_pre, 1.0):
                raise BacktestError(f"cash went negative ({cash_now:.4f}) on {index[t].date()}")

        # 3. book the bar
        units_hist[t] = units
        invested[t] = float((units * px[t]).sum())
        cash[t] = cash_now
        nav_t = cash[t] + invested[t]
        if not np.isfinite(nav_t):
            raise BacktestError(f"non-finite NAV on {index[t].date()}")

        # 4. decide at the close (history up to t only)
        if t in decide_at and t + config.execution_delay < n_bars:
            current = pd.Series(units * px[t] / nav_t if nav_t > 0 else 0.0, index=tickers)
            ctx = DecisionContext(date=index[t], history=prices.iloc[: t + 1],
                                  current_weights=current, nav=nav_t)
            raw = weight_fn(ctx)
            if raw is None:
                continue
            target = _clean_weights(raw, tickers)
            drift = float((target - current).abs().max())
            if decisions and drift < config.drift_threshold:
                continue
            pending[t] = target
            decisions.append({"decision_date": index[t],
                              "execution_date": index[t + config.execution_delay],
                              "max_drift": drift, **{f"w_{k}": v for k, v in target.items()}})

    nav = cash + invested
    # P&L identity: every change in NAV is explained by price moves on the units held
    # over the bar, interest earned and fees paid -- nothing else may move it.
    if n_bars > 1:
        explained = ((units_hist[:-1] * (px[1:] - px[:-1])).sum(axis=1)
                     + interest_bar[1:] - fees_bar[1:])
        gap = np.abs(np.diff(nav) - explained)
        worst = int(np.argmax(gap))
        if gap[worst] > IDENTITY_TOLERANCE * max(config.initial_capital, float(np.max(nav))):
            raise BacktestError(
                f"P&L identity broken on {index[worst + 1].date()}: unexplained {gap[worst]:.3e}")
    first_gap = abs(nav[0] - (config.initial_capital + interest_bar[0] - fees_bar[0]))
    if first_gap > IDENTITY_TOLERANCE * config.initial_capital:
        raise BacktestError(f"first bar does not start from the initial capital ({first_gap:.3e})")
    if not decisions:
        notes.append("the weighting rule never returned weights: the run stayed in cash")

    nav_s = pd.Series(nav, index=index, name="nav")
    weights = pd.DataFrame(units_hist * px, index=index, columns=tickers).div(nav_s, axis=0)
    return BacktestResult(
        nav=nav_s,
        cash=pd.Series(cash, index=index, name="cash"),
        invested=pd.Series(invested, index=index, name="invested"),
        weights=weights,
        units=pd.DataFrame(units_hist, index=index, columns=tickers),
        trades=pd.DataFrame(trades, columns=["date", "ticker", "units", "price", "notional", "fee"]),
        decisions=pd.DataFrame(decisions),
        config=config, label=label, fees_paid=float(fees_total), notes=notes, prices=prices,
    )



def _run_fast(prices: pd.DataFrame, weight_fn: WeightFn, config: BacktestConfig, label: str) -> BacktestResult:
    """Same rules as the bar-by-bar loop: between two bars where something happens (a decision or
    an order), positions are fixed and cash only earns interest, so those bars are filled at once."""
    prices = validate_prices(prices)
    if prices.isna().any().any():
        raise ValueError("the backtest needs a full panel; use data.common_window() first")
    index, tickers = prices.index, prices.columns
    dates, names = index.tolist(), list(tickers)          # boxed once, not at every trade
    px = prices.to_numpy()
    n_bars, n_assets = px.shape

    decide_at = decision_dates(index, config.rebalance, config.lookback)
    decide_set = set(decide_at)
    exec_at: dict[int, list[int]] = {}
    for d in decide_at:
        e = d + int(config.execution_delay)
        if e < n_bars:
            exec_at.setdefault(e, []).append(d)

    day_gap = np.r_[0.0, np.diff(index.to_numpy(dtype="datetime64[ns]")).astype("int64") / 86_400e9]
    if config.cash_prices is not None:
        level = config.cash_prices.sort_index()
        level = level[~level.index.duplicated(keep="last")].reindex(index, method="ffill")
        growth = (level / level.shift(1)).fillna(1.0).to_numpy()
    elif config.cash_rate:
        growth = (1.0 + config.cash_rate) ** (day_gap / 365.0)
        growth[0] = 1.0
    else:
        growth = None

    cash = np.empty(n_bars)
    units_hist = np.zeros((n_bars, n_assets))
    interest_bar = np.zeros(n_bars)
    fees_bar = np.zeros(n_bars)
    units = np.zeros(n_assets)
    cash_now = float(config.initial_capital)
    pending: dict[int, pd.Series] = {}
    trades, decisions, notes = [], [], []
    fees_total = 0.0

    events = sorted(set(decide_at) | set(exec_at) | {0, n_bars - 1})
    last = 0
    for b in events:
        # 1. the quiet bars up to b: fixed positions, cash earning interest
        if b > last or b == 0:
            s = last + 1 if b > 0 else 0
            span = np.arange(s, b + 1)
            if growth is not None and cash_now > 0 and len(span):
                path = cash_now * np.cumprod(growth[span])
                prev = np.r_[cash_now, path[:-1]]
                interest_bar[span] = path - prev
                cash[span] = path
                cash_now = float(path[-1])
            else:
                cash[span] = cash_now
            units_hist[span] = units
        t = b
        # 2. orders decided earlier
        live = [d for d in exec_at.get(t, ()) if d in pending]
        if live:
            target = pending.pop(max(live))
            for older in live:
                pending.pop(older, None)
            value_now = units * px[t]
            nav_pre = cash_now + value_now.sum()
            w = target.to_numpy()
            nav_post = nav_pre
            for _ in range(20):
                fee = config.fee_rate * np.abs(w * nav_post - value_now).sum()
                nav_post_new = nav_pre - fee
                if abs(nav_post_new - nav_post) < 1e-12 * max(nav_pre, 1.0):
                    nav_post = nav_post_new
                    break
                nav_post = nav_post_new
            new_units = w * nav_post / px[t]
            delta = new_units - units
            notional = delta * px[t]
            fee = config.fee_rate * np.abs(notional).sum()
            cash_now = cash_now - notional.sum() - fee
            fees_total += fee
            units = new_units
            for j in np.flatnonzero(np.abs(notional) > 1e-9):
                trades.append({"date": dates[t], "ticker": names[j], "units": delta[j],
                               "price": px[t, j], "notional": notional[j],
                               "fee": config.fee_rate * abs(notional[j])})
            fees_bar[t] += fee
            if cash_now < -1e-6 * max(nav_pre, 1.0):
                raise BacktestError(f"cash went negative ({cash_now:.4f}) on {index[t].date()}")
        # 3. book the bar
        units_hist[t] = units
        cash[t] = cash_now
        nav_t = cash_now + float((units * px[t]).sum())
        if not np.isfinite(nav_t):
            raise BacktestError(f"non-finite NAV on {index[t].date()}")
        last = t
        # 4. decide at the close
        if t in decide_set and t + config.execution_delay < n_bars:
            now_w = units * px[t] / nav_t if nav_t > 0 else np.zeros(n_assets)
            current = pd.Series(now_w, index=tickers)
            ctx = DecisionContext(date=dates[t], history=prices.iloc[: t + 1], current_weights=current, nav=nav_t)
            raw = weight_fn(ctx)
            if raw is None:
                continue
            target = _clean_weights(raw, tickers)
            drift = float(np.abs(target.to_numpy() - now_w).max())
            if decisions and drift < config.drift_threshold:
                continue
            pending[t] = target
            decisions.append({"decision_date": dates[t],
                              "execution_date": dates[t + config.execution_delay],
                              "max_drift": drift, **{f"w_{k}": v for k, v in target.items()}})

    invested = (units_hist * px).sum(axis=1)
    nav = cash + invested
    if n_bars > 1:
        explained = ((units_hist[:-1] * (px[1:] - px[:-1])).sum(axis=1)
                     + interest_bar[1:] - fees_bar[1:])
        gap = np.abs(np.diff(nav) - explained)
        worst = int(np.argmax(gap))
        if gap[worst] > IDENTITY_TOLERANCE * max(config.initial_capital, float(np.max(nav))):
            raise BacktestError(
                f"P&L identity broken on {index[worst + 1].date()}: unexplained {gap[worst]:.3e}")
    first_gap = abs(nav[0] - (config.initial_capital + interest_bar[0] - fees_bar[0]))
    if first_gap > IDENTITY_TOLERANCE * config.initial_capital:
        raise BacktestError(f"first bar does not start from the initial capital ({first_gap:.3e})")
    if not decisions:
        notes.append("the weighting rule never returned weights: the run stayed in cash")

    nav_s = pd.Series(nav, index=index, name="nav")
    return BacktestResult(
        nav=nav_s, cash=pd.Series(cash, index=index, name="cash"),
        invested=pd.Series(invested, index=index, name="invested"),
        weights=pd.DataFrame(units_hist * px, index=index, columns=tickers).div(nav_s, axis=0),
        units=pd.DataFrame(units_hist, index=index, columns=tickers),
        trades=pd.DataFrame(trades, columns=["date", "ticker", "units", "price", "notional", "fee"]),
        decisions=pd.DataFrame(decisions), config=config, label=label,
        fees_paid=float(fees_total), notes=notes, prices=prices,
    )
