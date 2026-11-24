"""Universes by wrapper and conversion to the investor's currency."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_assistant import builder, universes
from portfolio_assistant.market_data import fetch_prices


def _fake(series: dict):
    def download(ticker, start, end):
        s = series[ticker]
        return s.loc[start:] if start else s
    return download


def test_dollar_prices_are_divided_by_the_rate_of_the_same_day(tmp_path):
    idx = pd.bdate_range("2020-01-01", periods=6)
    usd = pd.Series([100.0, 101, 102, 103, 104, 105], idx)
    eur = pd.Series([50.0, 50, 51, 51, 52, 52], idx)
    rate = pd.Series([1.10, 1.12, 1.08, 1.09, 1.11, 1.10], idx)          # dollars per euro
    rate = rate.drop(idx[3])                                             # no rate that day
    out = fetch_prices(["SPY", "EXSA.DE"], start="2020-01-01", cache_dir=tmp_path, verbose=False,
                       downloader=_fake({"SPY": usd, "EXSA.DE": eur, "EURUSD=X": rate}),
                       currencies={"SPY": "USD", "EXSA.DE": "EUR"}, base_currency="EUR")
    assert idx[3] not in out.index                                       # dropped, never filled
    np.testing.assert_allclose(out["SPY"], (usd / rate).dropna())
    np.testing.assert_allclose(out["EXSA.DE"], eur.drop(idx[3]))


def test_without_currencies_nothing_is_converted(tmp_path):
    idx = pd.bdate_range("2020-01-01", periods=5)
    usd = pd.Series(np.linspace(100, 110, 5), idx)
    out = fetch_prices(["SPY"], start="2020-01-01", cache_dir=tmp_path, verbose=False,
                       downloader=_fake({"SPY": usd}))
    np.testing.assert_allclose(out["SPY"], usd)


@pytest.mark.parametrize("name", universes.names())
def test_every_preset_is_valid_in_every_universe(name):
    for preset, definition in builder.presets(universes.assets(name)).items():
        assert builder.validate(definition) == [], (name, preset)


def test_the_pea_has_no_refuge_but_cash_and_the_open_universe_has_bonds():
    pea = builder.presets(universes.assets("pea"))
    assert "refuge" not in pea["Momentum dual"].get("risque", {})
    assert {a["classe"] for a in universes.assets("pea")} <= {"actions monde", "actions USA", "actions Europe"}
    world = builder.presets(universes.assets("monde"))
    assert world["Momentum dual"]["risque"]["refuge"] == "EXHA.DE"


def test_recent_assets_are_left_out_of_the_long_history_universe():
    tickers = {a["ticker"] for a in universes.assets("pea")}
    assert "PAEEM.PA" not in tickers
    assert "PAEEM.PA" in {a["ticker"] for a in universes.assets("pea", include_recent=True)}


def test_the_demo_universe_gives_exactly_the_strategies_already_frozen():
    legacy = [{"ticker": "SPY", "classe": "actions"}, {"ticker": "EFA", "classe": "actions"},
              {"ticker": "EEM", "classe": "actions"}, {"ticker": "AGG", "classe": "obligations"},
              {"ticker": "GLD", "classe": "or"}]
    assert builder.presets(legacy) == builder.presets(universes.assets("demo"))


def test_unknown_universe_is_explained():
    with pytest.raises(ValueError, match="univers inconnu"):
        universes.get("crypto")
