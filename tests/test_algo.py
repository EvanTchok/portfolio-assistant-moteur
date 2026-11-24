"""Algorithmic mode: causal signals, next-close execution, stops, limits and accounting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_assistant import algo
from portfolio_assistant.execution import CostModel


def _walk(n=800, assets=4, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2014-01-01", periods=n)
    return pd.DataFrame(100 * np.cumprod(1 + rng.normal(2e-4, 0.012, (n, assets)), axis=0), index=idx,
                        columns=[f"A{i}" for i in range(assets)])


@pytest.mark.parametrize("signal", list(algo.SIGNALS))
def test_signals_never_look_ahead(signal):
    p = _walk()
    spec = algo.AlgoSpec(signal=signal, fast=20, slow=60, low=30 if signal == "rsi" else 1.5)
    full = algo.signals(p, spec)
    cut = algo.signals(p.iloc[:500], spec)
    for k in full:
        pd.testing.assert_frame_equal(full[k].iloc[:500], cut[k])


def test_a_breakout_then_a_crash_hits_the_stop_at_the_next_close():
    idx = pd.bdate_range("2020-01-01", periods=60)
    x = np.r_[np.full(30, 100.0), [111.0], np.full(5, 112.0), [99.0], np.full(23, 98.0)]
    p = pd.DataFrame({"X": x, "Y": np.full(60, 100.0)}, index=idx)
    spec = algo.AlgoSpec(signal="cassure", fast=20, slow=20, stop_loss=0.10, size=0.5, costs=CostModel(0.0))
    res = algo.run_algo(p, spec)
    row = res.journal.iloc[0]
    assert row["entrée"] == idx[31] and row["prix d'entrée"] == 112.0           # signal at 30, filled at 31
    assert row["motif"] == "stop-loss" and row["sortie"] == idx[37]             # crossed at 36, sold at 37
    assert row["résultat €"] == pytest.approx(0.5 * 100_000 / 112 * (98 - 112))


def test_an_unreached_limit_is_cancelled():
    idx = pd.bdate_range("2020-01-01", periods=60)
    x = np.r_[np.full(30, 100.0), np.linspace(111, 140, 30)]
    p = pd.DataFrame({"X": x, "Y": np.full(60, 100.0)}, index=idx)
    spec = algo.AlgoSpec(signal="cassure", fast=20, slow=20, order="limite", limit_offset=0.02, limit_days=3,
                         stop_loss=None)
    res = algo.run_algo(p, spec)
    assert "annulé" in set(res.orders["statut"]) and res.journal.empty         # the price never came back


def test_shorts_trailing_stops_and_costs_keep_the_books_balanced():
    p = _walk(seed=5)
    spec = algo.AlgoSpec(signal="croisement", fast=20, slow=60, short=True, trailing=0.08, take_profit=0.2,
                         max_gross=1.5, size=0.3, costs=CostModel(0.001, 1.0, spreads=True, slippage_bp=2))
    res = algo.run_algo(p, spec)
    assert (res.journal["sens"] == "vente à découvert").any()
    assert res.costs["commission"] > 0 and res.costs["borrow"] > 0
    s = res.stats()
    assert s["positions fermées"] > 5 and 0 <= s["gagnantes"] <= 1
    closed = res.journal[res.journal["statut"] == "close"]
    assert set(closed["motif"]) <= {"stop-loss", "objectif atteint", "stop suiveur", "durée maximale", "signal de sortie"}


def test_without_leverage_cash_never_goes_negative():
    p = _walk(assets=8, seed=2)
    res = algo.run_algo(p, algo.AlgoSpec(signal="rsi", fast=14, size=0.4, max_positions=8))
    assert res.cash.min() > -1e-6 and res.weights.sum(axis=1).max() <= 1 + 1e-9


def test_bad_settings_are_refused():
    with pytest.raises(ValueError):
        algo.AlgoSpec(signal="croisement", fast=200, slow=50)
    with pytest.raises(ValueError):
        algo.AlgoSpec(stop_loss=1.5)


def test_the_algorithmic_mode_runs_from_the_workshop(tmp_path):
    from portfolio_assistant import services, universes

    def prices(name, cache_dir=None, verbose=False):
        assets = universes.assets(name)
        rng = np.random.default_rng(len(name))
        idx = pd.bdate_range("2012-01-02", "2026-09-30")
        rets = rng.normal(3e-4, 0.01, (len(idx), len(assets)))
        return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=[a["ticker"] for a in assets])

    ws = services.Workspace(tmp_path, price_loader=prices)
    out = services.algo_run(ws, "monde", {"signal": "rsi", "fast": 14, "frais": "Courtier en ligne"},
                            services.atelier_instruments("monde"))
    assert out["period"][1] < "2023-01-01" and list(out["metrics"].columns)[0] == "Système"
    assert out["n_configurations"] == 1 and out["stats"]["positions fermées"] >= 0
    assert "Commissions" in set(out["costs"]["poste"])


def _ranged(x, lows=None, highs=None, opens=None):
    idx = pd.bdate_range("2020-01-01", periods=len(x))
    p = pd.DataFrame({"X": x, "Y": np.full(len(x), 100.0)}, index=idx)
    lo = p.copy(); hi = p.copy(); op = p.copy()
    for d, v in (lows or {}).items():
        lo.iloc[d, 0] = v
    for d, v in (highs or {}).items():
        hi.iloc[d, 0] = v
    for d, v in (opens or {}).items():
        op.iloc[d, 0] = v
    return p, {"open": op, "high": hi, "low": lo}


def test_a_limit_touched_inside_the_session_fills_at_its_limit():
    x = np.r_[np.full(30, 100.0), [111.0], np.full(29, 112.0)]
    spec = algo.AlgoSpec(signal="cassure", fast=20, slow=20, order="limite", limit_offset=0.02, limit_days=5,
                         stop_loss=None, costs=CostModel(0.0))
    p, r = _ranged(x, lows={32: 108.0})                                  # limit 108.78, close stays at 112
    assert algo.run_algo(p, spec).journal.empty                            # closes alone never reach it
    res = algo.run_algo(p, spec, ranges=r)
    assert res.journal.iloc[0]["prix d'entrée"] == pytest.approx(111 * 0.98)
    p2, r2 = _ranged(x, lows={32: 100.0}, opens={32: 105.0})              # opens below the limit
    assert algo.run_algo(p2, spec, ranges=r2).journal.iloc[0]["prix d'entrée"] == 105.0


def test_a_stop_touched_inside_the_session_exits_that_day():
    x = np.r_[np.full(30, 100.0), [111.0], np.full(29, 112.0)]
    spec = algo.AlgoSpec(signal="cassure", fast=20, slow=20, stop_loss=0.10, costs=CostModel(0.0))
    p, r = _ranged(x, lows={40: 95.0}, opens={40: 110.0})                # entry 112 → stop 100.8
    row = algo.run_algo(p, spec, ranges=r).journal.iloc[0]
    assert row["motif"] == "stop-loss (en séance)" and row["sortie"] == p.index[40]
    assert row["prix de sortie"] == pytest.approx(112 * 0.9)
    p, r = _ranged(x, lows={40: 90.0}, opens={40: 92.0})                 # gap below the stop: the open
    assert algo.run_algo(p, spec, ranges=r).journal.iloc[0]["prix de sortie"] == 92.0
