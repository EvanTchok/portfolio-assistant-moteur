"""Performance read on the NAV curve.

* Periods per year are measured on the index (observations / years covered), never
  assumed: daily data with holidays is not 252, weekly data is not 52 if weeks are missing.
* CAGR is geometric. The certainty equivalent (CRRA) says what a risk-averse client
  would accept for sure instead of this return stream.
* A metric computed on too few independent decisions is flagged, not hidden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def periods_per_year(index: pd.DatetimeIndex) -> float:
    index = pd.DatetimeIndex(index)
    if len(index) < 2:
        return float("nan")
    years = (index[-1] - index[0]).days / 365.25
    return float((len(index) - 1) / years) if years > 0 else float("nan")


def max_drawdown(nav: pd.Series) -> tuple[float, int]:
    """Deepest fall from a running peak, and the longest time under water (in bars)."""
    values = np.asarray(nav, dtype=np.float64)
    peak = np.maximum.accumulate(values)
    dd = values / peak - 1.0
    longest = current = 0
    for under in dd < -1e-12:
        current = current + 1 if under else 0
        longest = max(longest, current)
    return float(dd.min()), int(longest)


def certainty_equivalent(returns, gamma: float, ppy: float) -> float:
    """Annualised CRRA certainty equivalent. gamma=1 is the geometric mean."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size < 2 or np.any(r <= -1.0):
        return float("nan")
    if abs(gamma - 1.0) < 1e-12:
        per_period = np.exp(np.mean(np.log1p(r))) - 1.0
    else:
        per_period = np.mean((1.0 + r) ** (1.0 - gamma)) ** (1.0 / (1.0 - gamma)) - 1.0
    return float((1.0 + per_period) ** ppy - 1.0)


def summary(nav: pd.Series, *, risk_free: float = 0.0, n_decisions: int | None = None,
            gammas=(1.0, 3.0)) -> dict:
    nav = nav.dropna()
    if len(nav) < 3:
        raise ValueError("at least three NAV points are needed")
    r = nav.pct_change().iloc[1:].to_numpy()
    ppy = periods_per_year(nav.index)
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    total = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    cagr = float((1.0 + total) ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    vol = float(np.std(r, ddof=1) * np.sqrt(ppy))
    rf_period = (1.0 + risk_free) ** (1.0 / ppy) - 1.0
    excess = r - rf_period
    sharpe = float(np.mean(excess) / np.std(r, ddof=1) * np.sqrt(ppy)) if np.std(r) > 0 else float("nan")
    downside = np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2)) * np.sqrt(ppy)
    sortino = float(np.mean(excess) * ppy / downside) if downside > 0 else float("inf")
    mdd, underwater = max_drawdown(nav)
    out = {
        "start": nav.index[0], "end": nav.index[-1], "years": years,
        "periods_per_year": ppy, "total_return": total, "cagr": cagr,
        "volatility": vol, "sharpe": sharpe, "sortino": sortino,
        "max_drawdown": mdd, "max_underwater_bars": underwater,
        "calmar": cagr / abs(mdd) if mdd < 0 else float("inf"),
        "certainty_equivalent": {g: certainty_equivalent(r, g, ppy) for g in gammas},
        "warnings": [],
    }
    if n_decisions is not None:
        out["n_decisions"] = int(n_decisions)
        if n_decisions < 30:
            out["warnings"].append(
                f"only {n_decisions} rebalancing decisions: differences between strategies "
                "are not statistically resolvable on this sample")
    if years < 3:
        out["warnings"].append(f"{years:.1f} years of history: too short to judge a cycle")
    return out
