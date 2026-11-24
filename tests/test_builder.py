"""The block strategy builder: optimisers against closed forms, constraints, validation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import portfolio_assistant as pa
from portfolio_assistant import Lab
from portfolio_assistant import builder as b

TICKERS = ["SPY", "EFA", "EEM", "AGG", "GLD"]
CLASSES = ["actions", "actions", "actions", "obligations", "or"]
UNIVERSE = [{"ticker": t, "classe": c} for t, c in zip(TICKERS, CLASSES)]


def _prices(start="2010-01-01", end="2020-12-31", seed=0, vols=(0.012, 0.014, 0.018, 0.004, 0.010),
            drifts=(4e-4, 3e-4, 3e-4, 1e-4, 2e-4)):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    rets = rng.normal(drifts, vols, (len(idx), len(vols)))
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=TICKERS[: len(vols)])


def _frame(tickers, total=1.0, upper=1.0, caps=()):
    n = len(tickers)
    return b._Frame(list(tickers), np.zeros(n), np.full(n, upper), list(caps), total)


def _defn(**parts):
    return {"nom": "test", "univers": UNIVERSE, "rebalancement": "monthly", "frais": 0.001, **parts}


# ---------------------------------------------------------------------- catalogue
def test_the_catalogue_is_complete_and_serialisable():
    cat = b.catalogue()
    json.dumps(cat, ensure_ascii=False)
    for group in ("sélection", "pondération"):
        for block in cat[group]:
            assert block["libellé"] and len(block["description"]) > 20
    assert {"top_momentum", "trend", "absolute_momentum"} <= {x["id"] for x in cat["sélection"]}
    assert {"equal", "min_variance", "risk_parity", "max_sharpe", "fixed"} <= {x["id"] for x in cat["pondération"]}


def test_every_preset_is_valid():
    for name, defn in b.presets(UNIVERSE).items():
        assert b.validate(defn, available=TICKERS) == [], name


@pytest.mark.parametrize("change, message", [
    ({"selection": [{"type": "magic"}]}, "type inconnu"),
    ({"selection": [{"type": "top_momentum", "k": 0}]}, "doit être ≥"),
    ({"selection": [{"type": "trend", "days": 200, "speed": 3}]}, "réglage inconnu"),
    ({"ponderation": {"type": "fixed", "weights": {"TSLA": 1.0}}}, "hors univers"),
    ({"contraintes": {"max_per_class": {"crypto": 0.1}}}, "n'existe pas"),
    ({"contraintes": {"min_weight": 0.5, "max_weight": 0.2}}, "poids minimal dépasse"),
    ({"risque": {"refuge": "TLT"}}, "refuge"),
    ({"univers": []}, "au moins un actif"),
    ({"rebalancement": "daily"}, "rebalancement"),
])
def test_validation_names_the_problem_in_french(change, message):
    errors = b.validate(_defn(**change))
    assert any(message in e for e in errors), errors


def test_missing_price_data_is_reported():
    assert any("sans données" in e for e in b.validate(_defn(), available=["SPY", "AGG"]))


# ---------------------------------------------------------------------- optimisers
def test_min_variance_matches_the_closed_form_when_no_bound_binds():
    px = _prices(vols=(0.010, 0.012, 0.015), drifts=(3e-4, 3e-4, 3e-4))
    frame = _frame(px.columns)
    w = b._w_min_variance(px, frame, days=126)
    cov = b._covariance(px, 126)
    inv = np.linalg.solve(cov, np.ones(3))
    expected = inv / inv.sum()
    assert np.all(expected > 0)
    np.testing.assert_allclose(w, expected, atol=1e-4)


def test_risk_parity_equalises_risk_contributions():
    px = _prices()
    w = b._w_risk_parity(px, _frame(px.columns), days=126)
    cov = b._covariance(px, 126)
    contrib = w * (cov @ w)
    shares = contrib / contrib.sum()
    np.testing.assert_allclose(shares, np.full(5, 0.2), atol=1e-4)
    assert abs(w.sum() - 1) < 1e-9


def test_max_sharpe_matches_the_tangency_portfolio_when_no_bound_binds():
    px = _prices(vols=(0.010, 0.011, 0.012), drifts=(4e-4, 4e-4, 4e-4), seed=3)
    w = b._w_max_sharpe(px, _frame(px.columns), days=252)
    cov = b._covariance(px, 252) * 252
    mu = b._returns(px, 252).mean().to_numpy() * 252
    raw = np.linalg.solve(cov, mu)
    if np.all(raw > 0):
        np.testing.assert_allclose(w, raw / raw.sum(), atol=2e-3)
    sharpe = lambda x: (x @ mu) / np.sqrt(x @ cov @ x)  # noqa: E731
    assert sharpe(w) >= sharpe(np.full(3, 1 / 3)) - 1e-9


def test_caps_that_cannot_hold_the_whole_portfolio_leave_the_rest_in_cash():
    px = _prices()[["SPY", "EFA"]]
    w = b._w_equal(px, _frame(px.columns, upper=0.3))
    np.testing.assert_allclose(w, [0.3, 0.3], atol=1e-9)


def test_class_cap_is_respected_by_every_weighting():
    px = _prices()
    mask = np.array([c == "actions" for c in CLASSES])
    for fn in (b._w_equal, b._w_inverse_vol, b._w_min_variance, b._w_risk_parity, b._w_max_sharpe):
        w = fn(px, _frame(px.columns, caps=[(mask, 0.4)]))
        assert w[mask].sum() <= 0.4 + 1e-6, fn.__name__
        assert abs(w.sum() - 1) < 1e-6, fn.__name__


# --------------------------------------------------------------------- strategies
def _run(defn, px=None):
    px = _prices() if px is None else px
    spec = b.to_spec(defn)
    return pa.run_backtest(px[list(spec.universe)], b.composite(defn), spec.config())


def _targets(result):
    cols = [c for c in result.decisions.columns if c.startswith("w_")]
    out = result.decisions.set_index("decision_date")[cols].rename(columns=lambda c: c[2:])
    return out.fillna(0.0)


def test_constraints_hold_at_every_rebalancing():
    defn = _defn(ponderation={"type": "max_sharpe", "days": 252},
                 contraintes={"max_weight": 0.35, "max_per_class": {"actions": 0.5}, "min_cash": 0.1})
    targets = _targets(_run(defn))
    assert (targets.max(axis=1) <= 0.35 + 1e-6).all()
    assert (targets[["SPY", "EFA", "EEM"]].sum(axis=1) <= 0.5 + 1e-6).all()
    assert (targets.sum(axis=1) <= 0.9 + 1e-6).all()


def test_an_empty_selection_goes_to_the_refuge():
    defn = _defn(selection=[{"type": "max_volatility", "days": 63, "max_vol": 0.01}],  # nobody qualifies
                 risque={"refuge": "AGG"})
    targets = _targets(_run(defn))
    assert (targets["AGG"] == 1.0).all() and (targets.drop(columns="AGG") == 0).all().all()


def test_volatility_target_never_levers_and_caps_ex_ante_risk():
    px = _prices()
    defn = _defn(ponderation={"type": "equal"}, risque={"target_vol": 0.05, "days": 63})
    res = _run(defn, px)
    targets = _targets(res)
    assert (targets.sum(axis=1) <= 1 + 1e-9).all()
    for date, row in targets.iterrows():
        hist = px.loc[:date]
        cov = b._covariance(hist, 63) * 252
        w = row.to_numpy()
        assert np.sqrt(w @ cov @ w) <= 0.05 + 1e-9


def test_trend_filter_keeps_excluded_shares_in_cash():
    defn = b.presets(UNIVERSE)["Suivi de tendance"]
    targets = _targets(_run(defn))
    held = targets.gt(1e-9)
    np.testing.assert_allclose(targets.sum(axis=1), held.sum(axis=1) * 0.2, atol=1e-9)


def test_decisions_do_not_depend_on_the_future():
    px = _prices()
    altered = px.copy()
    altered.iloc[-300:] *= np.linspace(1, 3, 300)[:, None]      # rewrite the last 300 days
    defn = b.presets(UNIVERSE)["Tendance + parité de risque"]
    a, c = _targets(_run(defn, px)), _targets(_run(defn, altered))
    cut = px.index[-301]
    before = a.loc[a.index <= cut]
    pd.testing.assert_frame_equal(before, c.loc[before.index])


# ------------------------------------------------------------------------- the lab
def test_block_strategies_go_through_the_lab_and_grids_use_paths(tmp_path):
    lab = Lab(_prices(end="2022-12-30"), "2021-01-04", path=tmp_path / "research")
    defn = b.presets(UNIVERSE)["Momentum dual"]
    spec = b.to_spec(defn)
    lab.explore(spec, show=False)
    other = b.to_spec(b.set_path(defn, "selection.0.k", 2))
    assert lab.fingerprint(other) != lab.fingerprint(spec)
    lab.analyze(spec, grid={"selection.0.days": [126, 252]}, show=False)
    assert lab.n_trials == 2
    lab.freeze(spec, show=False)
    assert lab.test_oos(spec.name, show=False)["status"] == "VALID_OOS"


def test_definitions_round_trip_through_json(tmp_path):
    defn = b.presets(UNIVERSE)["Parité de risque"]
    path = b.save_definition(defn, tmp_path / "strategies")
    assert b.load_definition(path) == json.loads(json.dumps(defn))
    with pytest.raises(ValueError, match="invalide"):
        b.save_definition({**defn, "selection": [{"type": "nope"}]}, tmp_path / "strategies")


def test_describe_reads_like_a_sentence():
    text = b.describe(b.presets(UNIVERSE)["Tendance + parité de risque"])
    assert "Parité de risque" in text and "200" in text and "AGG" in text and "40%" in text


def test_optimisers_are_stable_under_invisible_price_noise():
    """Rounding prices at the 6th decimal must not change the strategy: a solver that
    stalls must not silently fall back to 1/N for the month."""
    px = _prices(start="2005-01-03", end="2026-10-02", vols=(0.012, 0.012, 0.012, 0.003, 0.011),
                 drifts=(3e-4,) * 5)
    defn = b.presets(UNIVERSE)["Markowitz"]
    fees = [_run(defn, p).fees_paid for p in (px, px.round(6))]
    assert abs(fees[1] / fees[0] - 1) < 1e-3, fees
