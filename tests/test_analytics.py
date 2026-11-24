"""Each analytic is checked against a closed form or an independent recomputation."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import portfolio_assistant as pa
from portfolio_assistant import analytics as an


def _prices(n_years=8, k=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=int(261 * n_years))
    px = 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, (len(idx), k)), axis=0)
    return pd.DataFrame(px, index=idx, columns=[f"A{i}" for i in range(k)])


def _nav(path, start="2020-01-01"):
    return pd.Series(path, index=pd.bdate_range(start, periods=len(path)), dtype=float)


def test_calendar_years_compound_to_the_total_return():
    nav = _prices()["A0"]
    years = an.calendar_returns(nav)
    total_from_years = float((1 + years["stratégie"]).prod() - 1)
    assert abs(total_from_years - (nav.iloc[-1] / nav.iloc[0] - 1)) < 1e-12


def test_monthly_grid_year_column_matches_the_calendar_year():
    nav = _prices()["A0"]
    grid, years = an.monthly_table(nav), an.calendar_returns(nav)
    full = years.index[~years["partielle"]]
    np.testing.assert_allclose(grid.loc[full, "total année"], years.loc[full, "stratégie"], atol=1e-12)


def test_historical_var_is_the_quantile_and_gaussian_var_agrees_on_normal_data():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0, 0.01, 200_000))
    v = an.value_at_risk(r, 0.95)
    assert abs(v["VaR historique"] + np.quantile(r, 0.05)) < 1e-15
    assert abs(v["VaR gaussienne"] - 0.01 * 1.6448536) < 2e-4
    assert abs(v["VaR Cornish-Fisher"] - v["VaR gaussienne"]) < 2e-4
    assert v["CVaR historique (perte moyenne au-delà)"] > v["VaR historique"]


def test_drawdown_table_reads_peak_trough_and_recovery():
    nav = _nav([100, 110, 99, 88, 95, 111, 120, 108, 130])
    table = an.drawdown_table(nav)
    first = table.iloc[0]
    assert abs(first["profondeur"] - (88 / 110 - 1)) < 1e-12
    assert first["sommet"] == nav.index[1].date() and first["creux"] == nav.index[3].date()
    assert first["retour au sommet"] == nav.index[5].date()
    assert abs(table.iloc[1]["profondeur"] - (108 / 120 - 1)) < 1e-12


def test_an_open_drawdown_is_reported_as_not_recovered():
    table = an.drawdown_table(_nav([100, 120, 90, 95]))
    assert table.iloc[0]["retour au sommet"] == "pas encore"


def test_relative_stats_of_a_series_against_itself():
    nav = _prices()["A0"]
    rel = an.relative_stats(nav, nav)
    assert abs(rel["bêta"] - 1) < 1e-12 and abs(rel["alpha annualisé (Jensen)"]) < 1e-12
    assert rel["tracking error"] < 1e-12 and abs(rel["corrélation"] - 1) < 1e-12


def test_beta_of_a_doubled_daily_return_is_two():
    bench = _prices()["A0"]
    levered = (1 + 2 * bench.pct_change().fillna(0)).cumprod() * 100
    assert abs(an.relative_stats(levered, bench)["bêta"] - 2) < 1e-9


def test_probabilistic_sharpe_matches_the_paper_formula():
    nav = _prices(n_years=10)["A1"]
    r = nav.pct_change().dropna().to_numpy()
    sr = r.mean() / r.std(ddof=1)
    g3, g4 = pd.Series(r).skew(), pd.Series(r).kurt() + 3
    expected = an.N01.cdf(sr * math.sqrt(len(r) - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2))
    stats = an.sharpe_statistics(nav)
    assert abs(stats["probabilité que le vrai Sharpe > 0 (PSR)"] - expected) < 1e-12
    assert stats["Sharpe - intervalle 95 % bas"] < stats["Sharpe annualisé"] < stats["Sharpe - intervalle 95 % haut"]


def test_more_trials_deflate_the_sharpe_more():
    nav = _prices(n_years=10)["A1"]
    psr = an.sharpe_statistics(nav)["probabilité que le vrai Sharpe > 0 (PSR)"]
    few = an.sharpe_statistics(nav, n_trials=5)["Sharpe dégonflé (DSR) - probabilité réelle"]
    many = an.sharpe_statistics(nav, n_trials=500)["Sharpe dégonflé (DSR) - probabilité réelle"]
    assert many < few < psr


def test_composition_shares_sum_to_one_and_follow_the_weights():
    px = _prices()
    res = pa.run_backtest(px, pa.equal_weight(), pa.BacktestConfig(rebalance="monthly"))
    comp = an.composition(res)
    assert abs(comp["part du rendement"].sum() - 1) < 1e-9
    assert abs(comp["part du risque"].sum() - 1) < 1e-9
    assert (abs(comp["poids moyen"] - 1 / 3) < 0.02).all()
    single = an.composition(pa.run_backtest(px, pa.fixed_weights({"A2": 1.0})))
    assert abs(single.loc["A2", "part du rendement"] - 1) < 1e-9


def test_equal_weight_has_as_many_effective_lines_as_assets():
    res = pa.run_backtest(_prices(k=4), pa.equal_weight(), pa.BacktestConfig(rebalance="daily",
                                                                               fee_rate=0.0))
    stats = an.portfolio_stats(res, start=res.nav.index[2])
    assert abs(stats["nombre effectif de lignes (1/HHI) moyen"] - 4) < 1e-6


def test_crises_outside_the_period_are_left_out():
    px = _prices(n_years=3)        # 2015-2017 only
    a = pa.run_backtest(px, pa.equal_weight())
    table = an.crisis_table(a.nav, a.nav)
    assert list(table["épisode"]) == ["Chine / pétrole 2015-16"]


def test_full_analysis_builds_every_table():
    px = _prices(n_years=12)
    cfg = pa.BacktestConfig(lookback=252)
    s = pa.run_backtest(px, pa.momentum_top_k(252, 2), cfg, label="momo")
    b = pa.run_backtest(px, pa.equal_weight(), cfg, label="1/N")
    out = an.full_analysis(s, b, n_trials=6)
    for key in ("rendement", "risque", "ratios", "relatif", "statistiques", "années", "mois",
                "drawdowns", "crises", "régimes", "composition", "portefeuille", "courbes"):
        assert key in out
    assert out["courbes"].iloc[0].eq(1.0).all()
    with pytest.raises(ValueError):
        an.relative_stats(s.nav, b.nav.iloc[:-1])
