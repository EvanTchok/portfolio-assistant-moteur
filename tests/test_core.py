"""Every test here can fail: each checks a number against a closed form or a
property that a wrong implementation would break."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import portfolio_assistant as pa


def _prices(n=600, k=3, seed=0, freq="B"):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.01, (n, k))
    px = 100.0 * np.cumprod(1.0 + rets, axis=0)
    return pd.DataFrame(px, index=pd.bdate_range("2018-01-01", periods=n),
                        columns=[f"A{i}" for i in range(k)])


# ----------------------------------------------------------------------------- data
def test_hole_inside_history_is_refused():
    px = _prices()
    px.iloc[100, 1] = np.nan
    with pytest.raises(pa.DataError, match="not .*forward-filled"):
        pa.validate_prices(px)


def test_late_start_is_not_a_hole_and_common_window_cuts_it():
    px = _prices()
    px.iloc[:50, 2] = np.nan
    clean = pa.validate_prices(px)
    window = pa.common_window(clean)
    assert window.index[0] == px.index[50] and not window.isna().any().any()


def test_duplicated_dates_and_non_positive_prices_are_refused():
    px = _prices()
    with pytest.raises(pa.DataError, match="duplicated"):
        pa.validate_prices(pd.concat([px.iloc[:10], px.iloc[9:20]]))
    px.iloc[5, 0] = 0.0
    with pytest.raises(pa.DataError, match="non-positive"):
        pa.validate_prices(px)


def test_portfolio_weights_from_quantities(tmp_path):
    px = _prices(k=2)
    path = tmp_path / "pf.csv"
    path.write_text("Symbole;Quantité\n".replace(";", ",") + "A0,10\nA1,30\n")
    holdings = pa.load_portfolio_csv(path)
    w = pa.weights_from_holdings(holdings, px)
    last = px.iloc[-1]
    expected = 10 * last["A0"] / (10 * last["A0"] + 30 * last["A1"])
    assert abs(w["A0"] - expected) < 1e-12 and abs(sum(w.values()) - 1) < 1e-12


# --------------------------------------------------------------------------- engine
def test_rule_never_sees_the_future_and_trades_one_bar_later():
    px = _prices()
    seen = []

    def spy(ctx):
        seen.append((ctx.date, ctx.history.index[-1]))
        return {"A0": 1.0}

    res = pa.run_backtest(px, spy, pa.BacktestConfig(rebalance="monthly"))
    assert seen and all(decided == last for decided, last in seen)
    for _, row in res.decisions.iterrows():
        pos = px.index.get_loc(row["decision_date"])
        assert row["execution_date"] == px.index[pos + 1]
    assert set(res.trades["date"]) <= set(res.decisions["execution_date"])


def test_same_bar_execution_is_refused():
    with pytest.raises(ValueError, match="look-ahead"):
        pa.BacktestConfig(execution_delay=0)


def test_buy_and_hold_matches_the_closed_form_without_fees():
    px = _prices()
    w = {"A0": 0.5, "A1": 0.3, "A2": 0.2}
    cfg = pa.BacktestConfig(rebalance="once", fee_rate=0.0, initial_capital=1_000.0)
    res = pa.run_backtest(px, pa.buy_and_hold(w), cfg)
    e = 1  # executed on bar 1
    expected = sum(1_000.0 * w[t] * px[t].iloc[e:] / px[t].iloc[e] for t in w)
    np.testing.assert_allclose(res.nav.iloc[e:].to_numpy(), expected.to_numpy(), rtol=1e-12)
    assert len(res.trades) == 3


def test_fees_are_solved_on_the_post_fee_nav():
    px = _prices()
    f = 0.002
    res = pa.run_backtest(px, pa.fixed_weights({"A0": 1.0}),
                          pa.BacktestConfig(rebalance="once", fee_rate=f, initial_capital=1_000.0))
    # from all cash: N' = C - f*N'  ->  N' = C / (1 + f), and nothing is left in cash
    assert abs(res.nav.iloc[1] - 1_000.0 / (1 + f)) < 1e-9
    assert abs(res.cash.iloc[1]) < 1e-9
    assert abs(res.fees_paid - 1_000.0 * f / (1 + f)) < 1e-9


def test_daily_equal_weight_return_is_the_cross_sectional_mean():
    px = _prices(n=200)
    res = pa.run_backtest(px, pa.equal_weight(), pa.BacktestConfig(rebalance="daily", fee_rate=0.0))
    nav_r = res.nav.pct_change().iloc[2:]
    mean_r = px.pct_change().iloc[2:].mean(axis=1)
    np.testing.assert_allclose(nav_r.to_numpy(), mean_r.to_numpy(), rtol=0, atol=1e-12)


def test_cash_earns_its_rate_on_calendar_days():
    px = _prices(n=300)
    res = pa.run_backtest(px, lambda ctx: {}, pa.BacktestConfig(cash_rate=0.03, initial_capital=100.0))
    days = (px.index[-1] - px.index[0]).days
    assert abs(res.nav.iloc[-1] - 100.0 * 1.03 ** (days / 365)) < 1e-9


def test_leverage_negative_weights_and_unknown_tickers_are_refused():
    px = _prices()
    for bad, msg in (({"A0": 0.7, "A1": 0.6}, "leverage"), ({"A0": -0.1}, "negative"),
                     ({"ZZZ": 1.0}, "unknown")):
        with pytest.raises(ValueError, match=msg):
            pa.run_backtest(px, pa.fixed_weights(bad))


def test_monthly_decisions_are_the_last_bar_of_each_month():
    idx = pd.bdate_range("2020-01-01", "2020-06-30")
    bars = pa.decision_dates(idx, "monthly", lookback=0)
    months = [idx[b].month for b in bars[1:]]
    assert all(idx[b + 1].month != idx[b].month for b in bars[1:])
    assert months == [1, 2, 3, 4, 5]


def test_drift_threshold_skips_small_rebalances():
    px = _prices()
    loose = pa.run_backtest(px, pa.equal_weight(), pa.BacktestConfig(drift_threshold=0.5))
    tight = pa.run_backtest(px, pa.equal_weight(), pa.BacktestConfig(drift_threshold=0.0))
    assert loose.n_decisions == 1 < tight.n_decisions
    assert loose.fees_paid < tight.fees_paid


# -------------------------------------------------------------------------- metrics
def test_certainty_equivalent_at_gamma_one_is_the_geometric_mean():
    r = np.array([0.10, -0.05, 0.07, -0.02])
    geo = np.exp(np.mean(np.log1p(r))) - 1
    assert abs(pa.certainty_equivalent(r, 1.0, 1.0) - geo) < 1e-12
    assert pa.certainty_equivalent(r, 5.0, 1.0) < pa.certainty_equivalent(r, 1.0, 1.0)


def test_drawdown_reads_the_trough_not_the_closing_level():
    nav = pd.Series([100, 90, 70, 85, 95, 101.0])
    mdd, under = pa.max_drawdown(nav)
    assert abs(mdd + 0.30) < 1e-12 and under == 4


def test_periods_per_year_is_measured_not_assumed():
    idx = pd.bdate_range("2019-01-01", "2023-12-31")
    assert 255 < pa.periods_per_year(idx) < 263
    weekly = pd.date_range("2019-01-04", periods=260, freq="W-FRI")
    assert 51 < pa.periods_per_year(weekly) < 53


# -------------------------------------------------------------------------- compare
def test_comparison_refuses_different_dates():
    px = _prices()
    a = pa.run_backtest(px, pa.equal_weight())
    b = pa.run_backtest(px.iloc[:-5], pa.equal_weight())
    with pytest.raises(ValueError, match="same dates"):
        pa.compare(a, b)


def test_a_strategy_identical_to_the_baseline_is_a_tie():
    px = _prices(n=2000)
    cfg = pa.BacktestConfig(rebalance="weekly")
    _, _, out = pa.run_with_baseline(px, pa.equal_weight(), cfg)
    assert out["verdict"] == "NO_REAL_DIFFERENCE"
    assert abs(out["table"].set_index("metric").loc["CAGR", "gap"]) < 1e-12


def test_few_decisions_are_flagged_insufficient():
    px = _prices(n=600)
    _, _, out = pa.run_with_baseline(px, pa.inverse_volatility(63),
                                     pa.BacktestConfig(rebalance="quarterly", lookback=63))
    assert out["verdict"] == "INSUFFICIENT_SAMPLE"
