"""Grid of settings: every combination run and counted, the best chosen on the first part only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_assistant import algo, grille, services, universes


def _walk(n=900, assets=5, seed=4):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n)
    return pd.DataFrame(100 * np.cumprod(1 + rng.normal(2e-4, 0.012, (n, assets)), axis=0), index=idx,
                        columns=[f"A{i}" for i in range(assets)])


def test_combinations_skip_impossible_pairs_and_are_capped():
    base = algo.AlgoSpec(signal="croisement", fast=20, slow=100)
    combos = grille.combinations(base, {"fast": [20, 50, 150], "slow": [100, 200]})
    assert len(combos) == 5                                            # (150, 100) is impossible
    with pytest.raises(ValueError, match="limitée"):
        grille.combinations(base, {"fast": list(range(5, 60)), "slow": list(range(100, 160))})
    with pytest.raises(ValueError, match="inconnus"):
        grille.combinations(base, {"lune": [1]})


def test_parallel_and_serial_runs_agree_and_the_best_is_chosen_on_part_a():
    p = _walk()
    base = algo.AlgoSpec(signal="croisement", fast=20, slow=100)
    grid = {"fast": [10, 20, 40], "stop_loss": [0.05, 0.1]}
    one = grille.run_grid(p, base, grid, workers=1)
    two = grille.run_grid(p, base, grid, workers=2)
    cols = ["fast", "stop_loss", "sharpe", "sharpe_a", "sharpe_b"]
    pd.testing.assert_frame_equal(one["table"][cols], two["table"][cols])
    t = one["table"]
    assert one["best"]["sharpe_a"] == t["sharpe_a"].max() and one["n"] == 6
    assert 0 <= one["dsr"] <= 1 and one["chance"] > 0 and 1 <= one["rank_b"] <= 6


def test_a_grid_counts_every_combination_in_the_workshop(tmp_path):
    def prices(name, cache_dir=None, verbose=False):
        assets = universes.assets(name)
        rng = np.random.default_rng(1)
        idx = pd.bdate_range("2012-01-02", "2026-09-30")
        rets = rng.normal(3e-4, 0.01, (len(idx), len(assets)))
        return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=[a["ticker"] for a in assets])

    ws = services.Workspace(tmp_path, price_loader=prices)
    out = services.grid_run(ws, "pea", {"signal": "rsi", "fast": 14, "frais": "Courtier en ligne"},
                            {"fast": [7, 14], "low": [25.0, 30.0]})
    assert out["n_configurations"] == 4 and out["period"][1] < "2023-01-01"
    again = services.grid_run(ws, "pea", {"signal": "rsi", "fast": 14, "frais": "Courtier en ligne"},
                              {"fast": [7, 14, 21], "low": [25.0, 30.0]})
    assert again["n_configurations"] == 6                              # only the new ones are added


def test_online_grids_are_smaller_and_use_one_core(monkeypatch):
    monkeypatch.setenv("PA_EN_LIGNE", "1")
    base = algo.AlgoSpec(signal="rsi", fast=14)
    with pytest.raises(ValueError, match="en ligne"):
        grille.combinations(base, {"fast": list(range(2, 40)), "low": [float(x) for x in range(10, 30)]})
    out = grille.run_grid(_walk(n=500), base, {"fast": [7, 14]})
    assert out["workers"] == 1
    assert grille.estimate_seconds(_walk(n=500), base, 10) > 0
