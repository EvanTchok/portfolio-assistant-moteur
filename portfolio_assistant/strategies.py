"""Weighting rules. Each one is a factory returning ``fn(ctx) -> weights``.

They only read ``ctx.history``, which stops at the decision date.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .engine import DecisionContext


def equal_weight(invested_fraction: float = 1.0):
    """1/N over every ticker. The reference any rule must beat."""
    def fn(ctx: DecisionContext):
        n = ctx.history.shape[1]
        return pd.Series(invested_fraction / n, index=ctx.history.columns)
    fn.__name__ = "equal_weight"
    return fn


def fixed_weights(weights: Mapping[str, float]):
    """The CGP's model allocation, rebalanced back on schedule."""
    target = {str(k).upper(): float(v) for k, v in weights.items()}
    def fn(ctx: DecisionContext):
        return target
    fn.__name__ = "fixed_weights"
    return fn


def buy_and_hold(weights: Mapping[str, float] | None = None):
    """Invest once, then never trade. Use with rebalance='once'."""
    inner = fixed_weights(weights) if weights is not None else equal_weight()
    state = {"done": False}
    def fn(ctx: DecisionContext):
        if state["done"]:
            return None
        state["done"] = True
        return inner(ctx)
    fn.__name__ = "buy_and_hold"
    return fn


def inverse_volatility(lookback: int = 63, invested_fraction: float = 1.0):
    """Weights proportional to 1 / realised volatility over ``lookback`` bars."""
    def fn(ctx: DecisionContext):
        rets = ctx.returns.iloc[-lookback:]
        if len(rets) < max(lookback // 2, 2):
            return None
        vol = rets.std(ddof=1).replace(0.0, np.nan)
        inv = 1.0 / vol
        if not np.isfinite(inv).any():
            return None
        inv = inv.fillna(0.0)
        return inv / inv.sum() * invested_fraction
    fn.__name__ = "inverse_volatility"
    return fn


def momentum_top_k(lookback: int = 252, k: int = 3, skip: int = 21):
    """Equal weight on the ``k`` best performers over ``lookback`` bars, skipping
    the last ``skip`` bars (short-term reversal). Cash if nothing is positive."""
    def fn(ctx: DecisionContext):
        hist = ctx.history
        if len(hist) <= lookback:
            return None
        end = hist.iloc[-1 - skip] if skip else hist.iloc[-1]
        start = hist.iloc[-1 - lookback]
        perf = (end / start - 1.0).sort_values(ascending=False)
        winners = perf[perf > 0.0].index[:k]
        if len(winners) == 0:
            return {}
        return {t: 1.0 / len(winners) for t in winners}
    fn.__name__ = "momentum_top_k"
    return fn


# Named rules a saved strategy can refer to. A strategy file stores the NAME and the
# parameters, never a Python object, so it can be reloaded months later.
RULES = {
    "equal_weight": equal_weight,
    "fixed_weights": fixed_weights,
    "buy_and_hold": buy_and_hold,
    "inverse_volatility": inverse_volatility,
    "momentum_top_k": momentum_top_k,
}
