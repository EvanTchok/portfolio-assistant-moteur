"""The fast path gives the reference loop's numbers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_assistant.engine import BacktestConfig, run_backtest
from portfolio_assistant.strategies import RULES, equal_weight, inverse_volatility, momentum_top_k


def _panel(seed, n=1500, k=6):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=n)
    return pd.DataFrame(100 * np.cumprod(1 + rng.normal(3e-4, 0.011, (n, k)), axis=0), index=idx,
                        columns=[f"A{i}" for i in range(k)])


CASES = [
    (equal_weight(), dict(rebalance="monthly")),
    (equal_weight(0.8), dict(rebalance="weekly", cash_rate=0.02)),
    (inverse_volatility(63), dict(rebalance="quarterly", lookback=63, fee_rate=0.003, drift_threshold=0.02)),
    (momentum_top_k(126, 2, 5), dict(rebalance="monthly", lookback=130, execution_delay=2, cash_rate=0.01)),
    (inverse_volatility(20, 0.6), dict(rebalance="daily", lookback=20, fee_rate=0.0005)),
    (equal_weight(), dict(rebalance="once")),
]


@pytest.mark.parametrize("seed", [1, 2])
@pytest.mark.parametrize("case", range(len(CASES)))
def test_fast_equals_reference(seed, case):
    rule, cfg = CASES[case]
    p = _panel(seed)
    a = run_backtest(p, rule, BacktestConfig(**cfg), reference=True)
    b = run_backtest(p, rule, BacktestConfig(**cfg))
    pd.testing.assert_series_equal(a.nav, b.nav, rtol=1e-10, atol=1e-7)
    pd.testing.assert_frame_equal(a.weights, b.weights, rtol=1e-9, atol=1e-9)
    assert a.fees_paid == pytest.approx(b.fees_paid, rel=1e-10)
    assert len(a.decisions) == len(b.decisions) and len(a.trades) == len(b.trades)


def test_fast_with_a_money_market_fund():
    p = _panel(3)
    fund = pd.Series(100 * 1.0001 ** np.arange(len(p)), p.index[::-1][::-1])
    cfg = BacktestConfig(rebalance="monthly", cash_prices=fund)
    a = run_backtest(p, equal_weight(0.7), cfg, reference=True)
    b = run_backtest(p, equal_weight(0.7), cfg)
    pd.testing.assert_series_equal(a.nav, b.nav, rtol=1e-10, atol=1e-7)
