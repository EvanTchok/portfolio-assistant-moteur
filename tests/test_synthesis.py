"""The research report reads the journals and produces a PDF."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from pypdf import PdfReader

from portfolio_assistant import Lab, Tracker, builder, universes
from portfolio_assistant.synthesis import build_report, collect
from portfolio_assistant.tracking import frozen_from_journal


def _prices(name, cache_dir=None, verbose=False):
    tickers = [a["ticker"] for a in universes.assets(name)]
    rng = np.random.default_rng(len(name))
    idx = pd.bdate_range("2012-01-02", "2026-09-30")
    vols = [0.003 if a["classe"] == "obligations" else 0.011 for a in universes.assets(name)]
    rets = rng.normal(3e-4, vols, (len(idx), len(tickers)))
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=tickers)


def test_the_report_is_built_from_the_journals(tmp_path):
    prices = _prices("monde")
    spec = builder.to_spec(builder.presets(universes.assets("monde"))["Variance minimale"])
    other = builder.to_spec(builder.presets(universes.assets("monde"))["Poids égaux"])
    lab = Lab(prices, "2023-01-01", path=tmp_path / "laboratoire" / "research_monde", profile="prudent")
    lab.explore(other, show=False)
    lab.freeze(spec, show=False)
    lab.test_oos(spec.name, show=False)
    tracker = Tracker(tmp_path / "laboratoire" / "suivi")
    tracker.start(frozen_from_journal(lab.log_file, spec.name, profile="prudent"),
                  label="Variance minimale · monde · prudent", source={"univers": "monde", "profil": "prudent"},
                  today=dt.date(2026, 9, 1))
    tracker.update("Variance minimale · monde · prudent", prices)

    data = collect(tmp_path)
    j = data["journals"]["monde"]
    assert len(j["trials"]) == 2 and len(j["freezes"]) == 1 and len(j["tests"]) == 1
    assert j["tests"][0]["status"] == "VALID_OOS" and j["tests"][0]["freeze"] is not None
    assert data["tracked"][0]["points"]

    path = build_report(tmp_path, tmp_path / "rapports" / "r.pdf", author="Test", price_loader=_prices)
    text = "".join(page.extract_text() for page in PdfReader(path).pages)
    for expected in ("Synthèse", "Variance minimale", "VALID OOS", "Prudent", "Suivi en temps réel",
                     "Limites connues", "journal complet"):
        assert expected in text, expected
    assert not list((tmp_path / "rapports").glob(".*_graphs"))       # temporary charts cleaned up


def test_an_empty_project_still_gives_a_report(tmp_path):
    path = build_report(tmp_path, tmp_path / "r.pdf", author="", price_loader=_prices)
    assert "Aucun test unique" in "".join(p.extract_text() for p in PdfReader(path).pages)


def test_the_report_speaks_french_figures_and_names():
    from portfolio_assistant.synthesis import _fr, _pct, track_title
    assert _pct(0.163) == "16,3 %" and _pct(-0.023, True) == "−2,3 %"
    assert _fr("Rebalancement : monthly, frais 0.10% par transaction") == \
        "Rebalancement : mensuel, frais 0,10 % par transaction"
    assert track_title("Suivi de tendance · monde · equilibre") == \
        "Suivi de tendance — Compte-titres / assurance-vie · Équilibré"
