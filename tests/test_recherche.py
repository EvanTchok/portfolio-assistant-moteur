"""Research mode and blind practice."""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_assistant import algo, recherche


def _walk(n=1500, assets=4, seed=2):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.DataFrame(100 * np.cumprod(1 + rng.normal(2e-4, 0.01, (n, assets)), axis=0), index=idx,
                        columns=[f"A{i}" for i in range(assets)])


def test_a_signal_followed_by_rises_shows_an_edge_and_a_random_one_does_not():
    p = _walk()
    spec = algo.AlgoSpec(signal="zscore", fast=20, low=1.5, stop_loss=None)
    rnd = recherche.study(p, spec, horizons=(5, 21))
    assert rnd["n"] > 20 and set(rnd["table"]["horizon"]) == {"5 j", "21 j"}
    # plant a rise after every entry signal: the study must see it
    sig = algo.signals(p, spec)["entry_long"]
    r = p.pct_change().fillna(0.0).to_numpy(copy=True)          # pandas 3: read-only otherwise
    for t, j in zip(*np.nonzero(sig.to_numpy())):
        r[t + 2: t + 7, j] += 0.004
    q = pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=p.index, columns=p.columns)
    rich = recherche.study(q, spec, horizons=(5,))
    assert rich["table"].loc[0, "verdict"] == "avance"


def test_first_touch_and_entry_at_the_next_close():
    idx = pd.bdate_range("2020-01-01", periods=60)
    x = np.r_[np.full(30, 100.0), [111.0, 112.0], np.linspace(112, 125, 10), np.full(18, 125.0)]
    p = pd.DataFrame({"X": x, "Y": np.full(60, 100.0)}, index=idx)
    out = recherche.study(p, algo.AlgoSpec(signal="cassure", fast=20, slow=20, stop_loss=None), horizons=(5,),
                          target=0.05, stop=0.05, window=20, n_boot=50)
    first = out["events"].iloc[0]
    assert first["date"] == idx[30] and first["5 j"] == x[36] / 112 - 1                     # from the close after the signal
    assert first["premier touché"] == "objectif"


def test_blind_practice_never_shows_the_future():
    p = _walk()
    b = recherche.Blind.draw(p, seed=3, shown=100, steps=10, step_bars=5)
    assert len(b.visible()) == 100
    b.decide(1)
    assert len(b.visible()) == 105 and b.position == 1
    for _ in range(9):
        b.decide(0)
    assert b.done and len(b.visible()) == 150
    s = b.score()
    assert s["pas"] == 10 and abs(s["vous"] - (b.nav[-1] / 100 - 1)) < 1e-12
