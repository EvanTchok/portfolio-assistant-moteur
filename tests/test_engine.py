"""Engine: remunerated cash."""

import pytest
def test_cash_earns_what_the_money_market_fund_earned_even_when_negative():
    import numpy as np
    import pandas as pd
    from portfolio_assistant.engine import BacktestConfig, run_backtest
    from portfolio_assistant.strategies import fixed_weights
    idx = pd.bdate_range("2020-01-01", periods=300)
    prices = pd.DataFrame({"A": np.full(len(idx), 100.0)}, idx)       # a flat asset: only cash moves
    for daily, sign in ((1e-4, 1), (-1e-4, -1)):
        fund = pd.Series(100 * (1 + daily) ** np.arange(len(idx)), idx)
        res = run_backtest(prices, fixed_weights({"A": 0.5}), BacktestConfig(fee_rate=0.0, cash_prices=fund))
        gain = res.nav.iloc[-1] / res.nav.iloc[0] - 1
        assert np.sign(gain) == sign
        assert (res.nav - res.cash - res.invested).abs().max() < 1e-6          # accounting identity holds
    # a fund price published AFTER the bar is never used (past prices only)
    late = pd.Series([100.0, 200.0], [idx[0], idx[-1] + pd.Timedelta(days=1)])
    res = run_backtest(prices, fixed_weights({"A": 0.5}), BacktestConfig(fee_rate=0.0, cash_prices=late))
    assert res.nav.iloc[-1] == pytest.approx(res.nav.iloc[0])
