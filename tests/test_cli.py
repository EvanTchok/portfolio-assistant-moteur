"""Command line: same workshop, same lab, same counting."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from portfolio_assistant import cli, services, universes


def _prices(name, cache_dir=None, verbose=False):
    assets = universes.assets(name)
    rng = np.random.default_rng(len(name))
    idx = pd.bdate_range("2012-01-02", "2026-09-30")
    rets = rng.normal(3e-4, 0.01, (len(idx), len(assets)))
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=[a["ticker"] for a in assets])


@pytest.fixture
def root(tmp_path, monkeypatch):
    real = services.Workspace
    monkeypatch.setattr(services, "Workspace", lambda r, price_loader=None: real(tmp_path, price_loader=_prices))
    monkeypatch.delenv("PA_EN_LIGNE", raising=False)
    return tmp_path


def run(capsys, *args):
    assert cli.main(["--json", *args]) == 0
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_new_try_explore_freeze(root, capsys):
    path = run(capsys, "nouvelle", "Ma stratégie")["fichier"]
    assert path.endswith("ma-strategie.py")
    tried = run(capsys, "essayer", path, "--univers", "monde")
    assert tried["période"][1] < "2023-01-01" and tried["configurations"] == 1
    explored = run(capsys, "explorer", path, "--univers", "monde")
    assert explored["nouvel essai"] and explored["essais"] == 1
    assert len(run(capsys, "geler", path, "--univers", "monde")["empreinte"]) == 12
    assert run(capsys, "essais")["monde"] == 1


def test_save_state_and_resume(root, capsys, tmp_path):
    path = run(capsys, "nouvelle", "Reprise")["fichier"]
    state = str(tmp_path / "etat.json")
    first = run(capsys, "essayer", path, "--univers", "monde", "--jusqu-au", "2018-12-31", "--etat", state)
    assert first["période"][1] <= "2018-12-31"
    rest = run(capsys, "reprendre", "--etat", state)
    full = run(capsys, "essayer", path, "--univers", "monde", "--jusqu-au", "2022-12-31", "--etat", str(tmp_path / "x.json"))
    assert rest["valeur finale"] == pytest.approx(full["valeur finale"], rel=1e-10)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n# changed\n")
    with pytest.raises(SystemExit, match="changé"):
        cli.main(["reprendre", "--etat", state])
