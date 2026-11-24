"""An adviser brings his own funds: a custom universe, its journal, its lock."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_assistant import builder, profiles, services, universes

FUNDS = [{"ticker": "aaa.pa", "nom": "Actions A", "classe": "actions", "devise": "EUR"},
         {"ticker": "BBB.DE", "nom": "Obligations B", "classe": "obligations", "devise": "EUR"},
         {"ticker": "CCC", "nom": "Or C", "classe": "or", "devise": "USD"}]


def _prices(name, cache_dir=None, verbose=False):
    assets = universes.assets(name)
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2012-01-02", "2026-09-30")
    rets = rng.normal(3e-4, 0.008, (len(idx), len(assets)))
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=[a["ticker"] for a in assets])


@pytest.fixture
def ws(tmp_path):
    w = services.Workspace(tmp_path, price_loader=_prices)
    yield w
    universes.use_folder("donnees_cabinet/univers")


def test_a_custom_universe_is_saved_listed_and_usable(ws):
    assert universes.validate_custom("cabinet", "Sélection du cabinet", FUNDS) == []
    universes.save_custom("cabinet", "Sélection du cabinet", FUNDS, root=ws.root)
    assert "cabinet" in universes.names()
    u = universes.get("cabinet")
    assert u["journal"] == "laboratoire/research_cabinet" and u["personnalisé"]
    assert [a["ticker"] for a in universes.assets("cabinet")] == ["AAA.PA", "BBB.DE", "CCC"]
    assert profiles.reference_weights(profiles.get("equilibre"), universes.assets("cabinet")) == \
        {"AAA.PA": 0.5, "BBB.DE": 0.5}
    d = builder.presets(universes.assets("cabinet"))["Variance minimale"]
    out = services.explore(ws, d, "cabinet", "equilibre")
    assert out["new_trial"] and (ws.root / "laboratoire" / "research_cabinet" / "registry.jsonl").exists()
    assert "cabinet" in ws.status()["journals"]


def test_a_universe_with_trials_is_locked(ws):
    universes.save_custom("cabinet", "Sélection du cabinet", FUNDS, root=ws.root)
    d = builder.presets(universes.assets("cabinet"))["Poids égaux"]
    services.explore(ws, d, "cabinet", None)
    assert universes.is_locked("cabinet", ws.root)
    with pytest.raises(ValueError, match="ne peuvent plus"):
        universes.save_custom("cabinet", "Sélection du cabinet", FUNDS[:2], root=ws.root)


def test_bad_universes_are_explained():
    errs = universes.validate_custom("monde", "", [FUNDS[1], {**FUNDS[1]}, {"ticker": "X", "classe": "crypto",
                                                                           "devise": "BTC"}])
    text = " ".join(errs)
    assert "univers fourni" in text and "libellé" in text and "double" in text
    assert "crypto" in text and "BTC" in text and "actions" in text
    assert universes.slug("Sélection AV — Cabinet Dupont") == "selection_av_cabinet_dupont"


def test_same_indices_count_as_consulted(ws):
    universes.save_custom("cabinet", "Sélection du cabinet", FUNDS, same_indices_as=["monde"], root=ws.root)
    assert universes.get("cabinet")["test_déjà_vu_via"] == ["monde"]
    assert any("déjà consultée" in r for r in universes.get("cabinet")["remarques"])


def test_prices_are_checked_before_saving():
    def fetch(t):
        if t == "CCC":
            raise ValueError("no data")
        return pd.Series([1.0, 2.0], pd.to_datetime(["2015-01-02", "2026-11-13"]))
    table = universes.check_prices(FUNDS, fetcher=fetch)
    assert table.loc["AAA.PA", "trouvé"] == "oui" and table.loc["CCC", "trouvé"].startswith("non")
