"""Client profiles: the reference portfolio, the judge, the limits, and the lab's bookkeeping."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_assistant import Lab, builder, profiles, universes

TICKERS = ["SPY", "EXSA.DE", "EXHA.DE", "GLD"]
UNIVERSE = [{"ticker": "SPY", "classe": "actions"}, {"ticker": "EXSA.DE", "classe": "actions"},
            {"ticker": "EXHA.DE", "classe": "obligations"}, {"ticker": "GLD", "classe": "or"}]


def _prices(end="2022-12-30", seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", end)
    rets = rng.normal([4e-4, 3e-4, 1e-4, 2e-4], [0.012, 0.013, 0.003, 0.010], (len(idx), 4))
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=TICKERS)


def _spec(name="Parité de risque"):
    return builder.to_spec(builder.presets(UNIVERSE)[name])


def test_profiles_are_ordered_from_cautious_to_bold():
    p, e, d = (profiles.get(n) for n in ("prudent", "equilibre", "dynamique"))
    assert p.gamma > e.gamma > d.gamma
    assert p.equity_share < e.equity_share < d.equity_share
    assert p.max_volatility < e.max_volatility < d.max_volatility
    assert p.max_loss < e.max_loss < d.max_loss


def test_reference_portfolio_splits_equities_and_bonds_and_ignores_the_rest():
    w = profiles.reference_weights(profiles.get("prudent"), UNIVERSE)
    assert w == pytest.approx({"SPY": 0.125, "EXSA.DE": 0.125, "EXHA.DE": 0.75})


def test_without_bonds_the_defensive_part_stays_in_cash():
    w = profiles.reference_weights(profiles.get("prudent"), universes.assets("pea"))
    assert sum(w.values()) == pytest.approx(0.25) and len(w) == 5


def test_a_lab_without_profile_is_unchanged(tmp_path):
    out = Lab(_prices(), "2020-01-01", path=tmp_path).explore(_spec(), show=False)
    assert out["reference"] == "1/N" and out["suitability"] == []


def test_a_profile_changes_the_reference_and_the_judge(tmp_path):
    lab = Lab(_prices(), "2020-01-01", path=tmp_path, profile="prudent")
    out = lab.explore(_spec(), show=False)
    strategy, baseline = out["results"]
    assert baseline.label == "référence Prudent"
    assert 6.0 in out["strategy"]["certainty_equivalent"]
    held = baseline.weights.iloc[-1]
    assert held["GLD"] == 0 and held["EXHA.DE"] > held["SPY"]


def test_a_strategy_beyond_the_limits_is_unsuitable_whatever_its_return(tmp_path):
    lab = Lab(_prices(), "2020-01-01", path=tmp_path, profile="prudent")
    out = lab.explore(_spec("Momentum dual"), show=False)       # all-in on one asset at a time
    assert out["strategy"]["volatility"] > 0.08
    assert out["verdict"] == "UNSUITABLE" and out["suitability"]


def test_one_strategy_judged_for_two_profiles_counts_as_two_trials(tmp_path):
    prices = _prices()
    Lab(prices, "2020-01-01", path=tmp_path, profile="prudent").explore(_spec(), show=False)
    lab = Lab(prices, "2020-01-01", path=tmp_path, profile="dynamique")
    lab.explore(_spec(), show=False)
    lab.explore(_spec(), show=False)                            # same look again: not recounted
    assert lab.n_trials == 2


def test_freeze_and_test_belong_to_a_profile_but_the_holdout_is_shared(tmp_path):
    prices = _prices()
    prudent = Lab(prices, "2020-01-01", path=tmp_path, profile="prudent")
    prudent.freeze(_spec(), show=False)
    dynamic = Lab(prices, "2020-01-01", path=tmp_path, profile="dynamique")
    with pytest.raises(ValueError, match="not frozen for this profile"):
        dynamic.test_oos(_spec().name, show=False)
    assert prudent.test_oos(_spec().name, show=False)["status"] == "VALID_OOS"
    dynamic.freeze(_spec(), show=False)                          # frozen AFTER the holdout was seen
    assert dynamic.test_oos(_spec().name, show=False)["status"] == "HOLDOUT_ALREADY_CONSULTED"


def test_robustness_is_measured_against_the_profile_reference(tmp_path):
    lab = Lab(_prices(), "2020-01-01", path=tmp_path, profile="equilibre")
    report = lab.analyze(_spec(), grid={"ponderation.days": [63, 126]}, show=False)
    rob = report["robustesse"]
    assert "écart CE γ=3 vs référence Équilibré" in rob["frais"].columns
    assert "écart CE γ=3 vs référence Équilibré" in set(rob["bootstrap"]["mesure"])
    assert any("battent référence Équilibré" in k for k in rob["paramètres - synthèse"].index)
    assert report["profil"]["name"] == "equilibre"
