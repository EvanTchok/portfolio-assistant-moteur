"""Advanced execution: same NAV as the lab's engine by default; costs, leverage, shorts and
margin calls each leave the trace they should."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_assistant import execution as ex
from portfolio_assistant.engine import BacktestConfig, run_backtest
from portfolio_assistant.strategies import equal_weight, momentum_top_k


def _prices(n=900, assets=5, seed=3, start="2015-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    rets = rng.normal(3e-4, 0.01, (n, assets))
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=[f"A{i}" for i in range(assets)])


@pytest.mark.parametrize("cash_rate", [0.0, 0.02])
def test_default_settings_give_the_lab_engine_nav(cash_rate):
    p = _prices()
    rule = momentum_top_k(lookback=126, k=2, skip=5)
    base = run_backtest(p, rule, BacktestConfig(lookback=130, cash_rate=cash_rate))
    adv = ex.run_advanced(p, rule, ex.AdvancedConfig(lookback=130, cash_rate=cash_rate))
    pd.testing.assert_series_equal(base.nav, adv.nav, rtol=1e-12, atol=1e-6)
    assert adv.result.fees_paid == pytest.approx(base.fees_paid)


def test_leverage_borrows_and_pays_for_it():
    p = _prices()
    cfg = ex.AdvancedConfig(leverage=2.0, max_gross=2.0, cash_rate=0.02)
    res = ex.run_advanced(p, equal_weight(), cfg)
    assert res.result.cash.min() < 0 and res.costs["debit"] > 0
    assert res.result.weights.abs().sum(axis=1).max() > 1.9
    with pytest.raises(ValueError):
        ex.run_advanced(p, equal_weight(), ex.AdvancedConfig(leverage=2.0))       # max_gross 1


def test_shorts_need_permission_and_pay_the_borrow():
    p = _prices()
    rule = ex.long_short_momentum(lookback=126, k=2, skip=5)
    with pytest.raises(ValueError, match="découvert"):
        ex.run_advanced(p, rule, ex.AdvancedConfig(lookback=130))
    res = ex.run_advanced(p, rule, ex.AdvancedConfig(lookback=130, allow_short=True))
    assert (res.result.weights < -0.1).any().any() and res.costs["borrow"] > 0


def test_a_crash_under_leverage_triggers_a_margin_call():
    idx = pd.bdate_range("2020-01-01", periods=60)
    path = np.r_[np.full(20, 100.0), np.linspace(100, 55, 10), np.full(30, 55.0)]
    p = pd.DataFrame({"X": path, "Y": 100 + np.zeros(60)}, index=idx)
    cfg = ex.AdvancedConfig(rebalance="once", leverage=3.0, max_gross=3.0, maintenance_margin=0.25)
    res = ex.run_advanced(p, lambda ctx: {"X": 1.0}, cfg)
    assert "appel de marge" in set(res.events["événement"]) or res.ruined
    assert res.result.trades["forcé"].ne("").any()


def test_a_loss_beyond_the_stake_is_a_ruin():
    idx = pd.bdate_range("2020-01-01", periods=30)
    p = pd.DataFrame({"X": np.r_[np.full(10, 100.0), np.full(20, 40.0)], "Y": np.full(30, 100.0)}, index=idx)
    res = ex.run_advanced(p, lambda ctx: {"X": 1.0}, ex.AdvancedConfig(rebalance="once", leverage=2.0, max_gross=2.0))
    assert res.ruined and res.result.units.iloc[-1].abs().sum() == 0


def test_french_transaction_tax_follows_its_history():
    assert ex.ttf_rate("2011-12-31") == 0 and ex.ttf_rate("2013-06-01") == 0.002
    assert ex.ttf_rate("2020-01-01") == 0.003 and ex.ttf_rate("2026-01-01") == 0.004
    m = ex.CostModel(0.0, taxes=True)
    shares = [ex.Instrument("MC.PA", "action", ttf=True), ex.Instrument("CW8.PA", "etf")]
    out = m.line_costs(np.array([10_000.0, 10_000.0]), shares, np.zeros(2), "2026-03-02")
    assert out["tax"].tolist() == [40.0, 0.0]
    sold = m.line_costs(np.array([-10_000.0, 0.0]), shares, np.zeros(2), "2026-03-02")
    assert sold["tax"].sum() == 0                                   # purchases only


def test_minimum_commission_and_impact():
    m = ex.CostModel(0.001, 5.0, impact_k=0.1)
    inst = [ex.Instrument("A", "etf", adv=1e6)]
    small = m.line_costs(np.array([1000.0]), inst, np.array([0.01]), "2020-01-01")
    assert small["commission"][0] == 5.0
    big = m.line_costs(np.array([1e6]), inst, np.array([0.01]), "2020-01-01")
    assert big["impact"][0] == pytest.approx(1e6 * 0.1 * 0.01 * 1.0)


def test_realistic_costs_cost_more_than_the_flat_fee():
    p = _prices()
    rule = momentum_top_k(lookback=126, k=2, skip=5)
    flat = ex.run_advanced(p, rule, ex.AdvancedConfig(lookback=130))
    bank = ex.run_advanced(p, rule, ex.AdvancedConfig(lookback=130, costs=ex.COST_PRESETS["Banque de réseau"]))
    assert bank.nav.iloc[-1] < flat.nav.iloc[-1]
    table = ex.cost_table(bank)
    assert {"Commissions", "Écart achat-vente"} <= set(table["poste"])


def test_cfd_and_forex_kinds():
    p = _prices(assets=2)
    p.columns = ["EURUSD=X", "DAX"]
    res = ex.run_advanced(p, lambda ctx: {"EURUSD=X": 0.5, "DAX": 0.5}, ex.AdvancedConfig(rebalance="quarterly"),
                          instruments={"DAX": "cfd"})
    assert res.costs["cfd"] > 0
    assert ex.guess_instrument("EURUSD=X").kind == "forex"


def test_the_workshop_runs_and_counts_its_own_configurations(tmp_path):
    from portfolio_assistant import builder, services, universes

    def prices(name, cache_dir=None, verbose=False):
        assets = universes.assets(name)
        rng = np.random.default_rng(len(name))
        idx = pd.bdate_range("2012-01-02", "2026-09-30")
        rets = rng.normal(3e-4, 0.01, (len(idx), len(assets)))
        return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=[a["ticker"] for a in assets])

    ws = services.Workspace(tmp_path, price_loader=prices)
    choices = services.atelier_choices(ws, "monde", "equilibre")
    trials = ws.lab("monde", "equilibre").n_trials
    one = services.atelier_run(ws, "monde", "equilibre", choices["Modèle — Poids égaux"],
                               {"frais": "Courtier en ligne", "levier": 1.5})
    again = services.atelier_run(ws, "monde", "equilibre", choices["Modèle — Poids égaux"],
                                 {"frais": "Courtier en ligne", "levier": 1.5})
    ls = next(k for k in choices if k.startswith("Atelier — Momentum"))
    third = services.atelier_run(ws, "monde", "equilibre", choices[ls], {"frais": "Institutionnel"})
    assert one["n_configurations"] == again["n_configurations"] == 1 and third["n_configurations"] == 2
    assert ws.lab("monde", "equilibre").n_trials == trials                  # the lab's count is untouched
    assert one["period"][1] < "2023-01-01" and one["curves"].index.max() < pd.Timestamp("2023-01-01")
    assert "Intérêts d'emprunt (levier)" in set(one["costs"]["poste"])


def test_conversion_fee_only_on_foreign_lines():
    m = ex.CostModel(0.0, fx_fee=0.0025)
    lines = [ex.Instrument("SPY", currency="USD"), ex.Instrument("CW8.PA")]
    out = m.line_costs(np.array([10_000.0, -10_000.0]), lines, np.zeros(2), "2020-01-01")
    assert out["fx"].tolist() == [25.0, 0.0]


def test_management_and_performance_fees():
    idx = pd.bdate_range("2020-01-01", "2022-12-31")
    p = pd.DataFrame({"X": 100 * 1.0004 ** np.arange(len(idx)), "Y": np.full(len(idx), 100.0)}, index=idx)
    rule = lambda ctx: {"X": 1.0}
    base = ex.run_advanced(p, rule, ex.AdvancedConfig(rebalance="once", costs=ex.CostModel(0.0)))
    mgmt = ex.run_advanced(p, rule, ex.AdvancedConfig(rebalance="once", costs=ex.CostModel(0.0), management_fee=0.02))
    years = (idx[-1] - idx[0]).days / 365
    assert mgmt.nav.iloc[-1] / base.nav.iloc[-1] == pytest.approx(0.98 ** years, rel=2e-3)
    perf = ex.run_advanced(p, rule, ex.AdvancedConfig(rebalance="once", costs=ex.CostModel(0.0), performance_fee=0.2))
    assert perf.costs["performance"] > 0
    net, hwm = 100_000.0, 100_000.0                                   # same thing by hand, year by year
    ends = base.nav.groupby(base.nav.index.year).last()
    previous = 100_000.0
    for level in ends:
        gross = net * level / previous
        net = gross - 0.2 * max(gross - hwm, 0.0)
        hwm, previous = max(hwm, net), level
    assert perf.nav.iloc[-1] == pytest.approx(net, rel=1e-9)


def test_a_portfolio_stop_goes_to_cash_at_the_next_close_and_waits():
    idx = pd.bdate_range("2020-01-01", periods=80)
    x = np.r_[np.full(25, 100.0), np.linspace(100, 80, 10), np.full(45, 80.0)]
    p = pd.DataFrame({"X": x, "Y": np.full(80, 100.0)}, index=idx)
    cfg = ex.AdvancedConfig(rebalance="quarterly", costs=ex.CostModel(0.0), triggers=(ex.Trigger("stop", 0.10),))
    res = ex.run_advanced(p, lambda ctx: {"X": 1.0}, cfg)
    fired = res.events[res.events["événement"].str.startswith("déclencheur")]
    assert len(fired) == 1
    when = idx.get_loc(fired["date"].iloc[0])
    assert x[when] / 100 - 1 <= -0.10 < x[when - 1] / 100 - 1                    # seen at the close where it crossed
    assert res.result.weights["X"].iloc[when] > 0.99 and res.result.weights["X"].iloc[when + 1] == 0   # sold next close
    assert "stop" in set(res.result.decisions["motif"])


def test_drift_and_volatility_triggers():
    p = _prices(n=600, assets=4, seed=9)
    cfg = ex.AdvancedConfig(rebalance="yearly", lookback=5, triggers=(ex.Trigger("derive", 0.03),))
    res = ex.run_advanced(p, equal_weight(), cfg)
    assert (res.result.decisions["motif"] == "derive").sum() >= 1
    cfg = ex.AdvancedConfig(rebalance="yearly", lookback=5, triggers=(ex.Trigger("volatilite", 0.05),))
    res = ex.run_advanced(p, equal_weight(), cfg)
    assert (res.result.decisions["motif"] == "volatilite").sum() >= 1
    with pytest.raises(ValueError):
        ex.Trigger("lune", 0.1)


def test_the_workshop_passes_fees_triggers_and_currencies(tmp_path):
    from portfolio_assistant import services, universes

    def prices(name, cache_dir=None, verbose=False):
        assets = universes.assets(name)
        rng = np.random.default_rng(3)
        idx = pd.bdate_range("2012-01-02", "2026-09-30")
        rets = rng.normal(3e-4, 0.012, (len(idx), len(assets)))
        return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=[a["ticker"] for a in assets])

    ws = services.Workspace(tmp_path, price_loader=prices)
    ch = services.atelier_choices(ws, "monde", None)
    out = services.atelier_run(ws, "monde", None, ch["Modèle — Poids égaux"],
                               {"frais": "Courtier en ligne", "frais de gestion": 0.01, "commission de performance": 0.1,
                                "déclencheurs": [{"type": "derive", "seuil": 0.02}]})
    posts = set(out["costs"]["poste"])
    assert {"Change de devises", "Frais de gestion"} <= posts                 # SPY & co are quoted in dollars
    assert out["events"]["événement"].str.startswith("déclencheur").any()


@pytest.mark.parametrize("settings", [
    {},
    {"leverage": 1.5, "max_gross": 1.5, "costs": ex.CostModel(0.001, 1.0, spreads=True, impact_k=0.1),
     "management_fee": 0.01, "performance_fee": 0.1, "triggers": (ex.Trigger("derive", 0.03), ex.Trigger("stop", 0.08))},
])
def test_a_run_stopped_and_resumed_is_the_same_run(settings):
    p = _prices(n=1200, assets=5, seed=12)
    rule = momentum_top_k(lookback=126, k=2, skip=5)
    cfg = ex.AdvancedConfig(lookback=130, cash_rate=0.01, **settings)
    full = ex.run_advanced(p, rule, cfg)
    cut = p.index[700]
    first = ex.run_advanced(p.loc[:cut], rule, cfg)
    saved = ex.RunState.from_dict(__import__("json").loads(__import__("json").dumps(first.state.to_dict())))
    rest = ex.run_advanced(p, rule, cfg, resume=saved)
    assert rest.nav.index[0] == p.index[701]
    pd.testing.assert_series_equal(rest.nav, full.nav.loc[p.index[701]:], rtol=1e-10, atol=1e-6)
    assert rest.state.totals["commission"] == pytest.approx(full.state.totals["commission"])
    with pytest.raises(ValueError):
        ex.run_advanced(p.loc[:cut], rule, cfg, resume=saved)                 # nothing new to run
