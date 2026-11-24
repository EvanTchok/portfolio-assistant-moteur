"""Strategy vs reference. A result is never shown alone.

The baseline uses the SAME prices, dates, rebalance calendar, delay and costs, so the
gap measures the weighting rule and nothing else.
"""

from __future__ import annotations

import pandas as pd

from .engine import BacktestConfig, BacktestResult, run_backtest
from .metrics import summary
from .strategies import equal_weight


def run_with_baseline(prices: pd.DataFrame, weight_fn, config: BacktestConfig | None = None,
                      label: str = "strategy", baseline_fn=None,
                      baseline_label: str = "equal weight (1/N)"):
    config = config or BacktestConfig()
    strategy = run_backtest(prices, weight_fn, config, label=label)
    baseline = run_backtest(prices, baseline_fn or equal_weight(), config, label=baseline_label)
    return strategy, baseline, compare(strategy, baseline)


def _decisions_from(result: BacktestResult, start) -> int:
    if result.decisions.empty:
        return 0
    return int((result.decisions["execution_date"] >= start).sum())


def _fees_from(result: BacktestResult, start) -> float:
    if result.trades.empty:
        return 0.0
    return float(result.trades.loc[result.trades["date"] >= start, "fee"].sum())


# Below this gap in certainty equivalent (per year), the two portfolios are called equal:
# a 0.05 %/year edge is far inside the noise of any backtest and would not survive a
# change of broker. A convention, like the robustness pass marks: change it if you want.
MATERIALITY = 0.0025


def compare(strategy: BacktestResult, baseline: BacktestResult, *, risk_free: float = 0.0,
            gamma: float = 3.0, start=None, materiality: float = MATERIALITY) -> dict:
    """``start`` restricts the measurement to a window (e.g. the out-of-sample period)."""
    if not strategy.nav.index.equals(baseline.nav.index):
        raise ValueError("strategy and baseline are not on the same dates: the gap would be meaningless")
    # Measure from the first day BOTH portfolios are invested: a warm-up year spent in
    # cash by both sides would only dilute the gap.
    starts = [r.decisions["execution_date"].iloc[0] for r in (strategy, baseline)
              if not r.decisions.empty]
    auto = max(starts) if starts else strategy.nav.index[0]
    start = max(auto, pd.Timestamp(start)) if start is not None else auto
    n_s, n_b = _decisions_from(strategy, start), _decisions_from(baseline, start)
    s = summary(strategy.nav.loc[start:], risk_free=risk_free, n_decisions=n_s, gammas=(1.0, gamma))
    b = summary(baseline.nav.loc[start:], risk_free=risk_free, n_decisions=n_b, gammas=(1.0, gamma))
    rows = [
        ("CAGR", "cagr", True), ("volatility", "volatility", False),
        ("Sharpe", "sharpe", True), ("max drawdown", "max_drawdown", True),
        ("Calmar", "calmar", True),
    ]
    table = pd.DataFrame(
        [(name, s[key], b[key], s[key] - b[key]) for name, key, _ in rows]
        + [(f"certainty equivalent γ={gamma:g}", s["certainty_equivalent"][gamma],
            b["certainty_equivalent"][gamma],
            s["certainty_equivalent"][gamma] - b["certainty_equivalent"][gamma])]
        + [("fees paid", _fees_from(strategy, start), _fees_from(baseline, start),
            _fees_from(strategy, start) - _fees_from(baseline, start)),
           ("rebalancing decisions", n_s, n_b, n_s - n_b)],
        columns=["metric", strategy.label, baseline.label, "gap"],
    )
    ce_gap = s["certainty_equivalent"][gamma] - b["certainty_equivalent"][gamma]
    if n_s < 30:
        verdict = "INSUFFICIENT_SAMPLE"
        reason = (f"only {n_s} rebalancing decisions in the window: the gap "
                  f"({ce_gap:+.2%}/year certain) is not statistically resolvable")
    elif abs(ce_gap) < materiality:
        verdict = "NO_REAL_DIFFERENCE"
        reason = (f"the gap ({ce_gap:+.2%}/year certain) is below the materiality threshold "
                  f"({materiality:.2%}/year): call it a tie")
    elif ce_gap > 0:
        verdict = "ADDS_VALUE"
        reason = f"a client with risk aversion γ={gamma:g} prefers it (+{ce_gap:.2%}/year certain)"
    else:
        verdict = "ADDS_NOTHING"
        reason = f"a client with risk aversion γ={gamma:g} prefers the baseline ({ce_gap:+.2%}/year)"
    return {"table": table, "strategy": s, "baseline": b, "verdict": verdict,
            "verdict_reason": reason, "start": start, "ce_gap": ce_gap}
