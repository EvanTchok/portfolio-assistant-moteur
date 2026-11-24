"""The twelve validation tests: construction period only, never a trial, one report per strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_assistant import builder, services, universes
from portfolio_assistant import validation_tests as vt


def _prices(name, cache_dir=None, verbose=False):
    assets = universes.assets(name)
    rng = np.random.default_rng(len(name))
    idx = pd.bdate_range("2012-01-02", "2026-09-30")
    vols = [0.003 if a["classe"] == "obligations" else 0.011 for a in assets]
    rets = rng.normal(3e-4, vols, (len(idx), len(assets)))
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=[a["ticker"] for a in assets])


def _reservoir(cache_dir, start="2005-01-01", base_currency="EUR"):
    idx = pd.bdate_range("2012-01-02", "2026-09-30")
    rng = np.random.default_rng(7)
    classes = {f"A{i}": "actions" for i in range(8)} | {f"O{i}": "obligations" for i in range(5)} \
        | {"G0": "or", "I0": "immobilier", "M0": "matières premières"}
    vols = [0.003 if c == "obligations" else 0.011 for c in classes.values()]
    rets = rng.normal(2e-4, vols, (len(idx), len(classes)))
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=list(classes)), classes


@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.setattr(vt, "reservoir_prices", _reservoir)
    return services.Workspace(tmp_path, price_loader=_prices)


def test_the_battery_gives_twelve_verdicts_and_counts_no_trial(ws):
    d = builder.presets(universes.assets("monde"))["Variance minimale"]
    services.freeze(ws, d, "monde", "prudent")
    before = ws.lab("monde", "prudent").n_trials
    out = services.validate(ws, "monde", "prudent", name=d["nom"], n_baskets=4)
    report = out["report"]
    assert [t["n"] for t in report["tests"]] == list(range(1, 13))
    assert all(t["verdict"] in ("passe", "limite", "échoue", "non applicable") for t in report["tests"])
    assert sum(report["synthèse"][k] for k in ("passés", "limites", "échoués", "non applicables")) == 12
    assert ws.lab("monde", "prudent").n_trials == before
    assert services.last_validation(ws, d["nom"], "monde", "prudent")["empreinte"] == report["empreinte"]


def test_the_sealed_period_stays_closed(ws):
    d = builder.presets(universes.assets("monde"))["Suivi de tendance"]
    services.freeze(ws, d, "monde", "equilibre")
    out = services.validate(ws, "monde", "equilibre", name=d["nom"], n_baskets=3)
    t11 = out["report"]["tests"][10]
    assert t11["verdict"] == "non applicable"                    # no test yet: nothing to read
    assert services.frozen_list(ws).iloc[0]["statut"] == "pas encore testée"


def test_an_unknown_strategy_is_refused(ws):
    with pytest.raises(ValueError):
        services.validate(ws, "monde", "prudent", name="Inexistante")


def test_french_inflation_is_shipped():
    s = vt.inflation()
    assert s.index.min() <= pd.Timestamp("2005-01-31") and len(s) > 200 and (s > 0).all()


def test_numbers_are_written_the_french_way():
    assert vt._pct(0.1234) == "12,3 %" and vt._pct(-0.05, 1, True).startswith("\u22125,0")
