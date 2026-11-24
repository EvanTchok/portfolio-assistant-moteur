"""Live tracking: only frozen strategies, only days after the start, everything on record."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from portfolio_assistant import Lab, Tracker, builder
from portfolio_assistant.tracking import frozen_from_journal

TICKERS = ["SPY", "EFA", "EEM", "AGG", "GLD"]
UNIVERSE = [{"ticker": t, "classe": c} for t, c in
            zip(TICKERS, ["actions", "actions", "actions", "obligations", "or"])]


def _prices(end="2020-12-31", seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", end)
    rets = rng.normal(3e-4, [0.012, 0.014, 0.018, 0.004, 0.010], (len(idx), 5))
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=TICKERS)


@pytest.fixture
def frozen(tmp_path):
    lab = Lab(_prices(end="2019-12-31"), "2018-01-01", path=tmp_path / "research")
    spec = builder.to_spec(builder.presets(UNIVERSE)["Parité de risque"])
    lab.freeze(spec, show=False)
    return frozen_from_journal(lab.log_file, spec.name)


def test_only_a_frozen_strategy_can_be_tracked(tmp_path, frozen):
    with pytest.raises(ValueError, match="gelée"):
        frozen_from_journal(tmp_path / "research" / "registry.jsonl", "Markowitz")


def test_the_track_starts_the_next_trading_day_and_counts_nothing_before(tmp_path, frozen):
    tracker = Tracker(tmp_path / "suivi")
    record = tracker.start(frozen, today=dt.date(2020, 6, 26))          # a Friday
    assert record["premier jour compté"] == "2020-06-29"                 # the Monday
    px = _prices()
    early = tracker.update(frozen["name"], px.loc[:"2020-06-26"])
    assert early["jours de bourse suivis"] == 0 and "message" in early

    out = tracker.update(frozen["name"], px)
    growth = out["courbes (base 100)"]
    assert growth.index[0] == pd.Timestamp("2020-06-26") and (growth.iloc[0] == 100).all()
    assert out["jours de bourse suivis"] == len(px.loc["2020-06-29":])
    expected = growth.iloc[-1, 0] / 100 - 1
    assert out["performance"].loc["rendement depuis le début", frozen["name"]] == pytest.approx(expected)
    assert out["statut"] == "INTACT" and not out["assez de recul"]


def test_starting_twice_is_harmless_but_a_new_configuration_is_refused(tmp_path, frozen):
    tracker = Tracker(tmp_path / "suivi")
    first = tracker.start(frozen, today=dt.date(2020, 6, 26))
    assert tracker.start(frozen, today=dt.date(2020, 9, 1)) == first      # the start date never moves
    other = {**frozen, "spec": {**frozen["spec"], "fee_rate": 0.005}}
    with pytest.raises(ValueError, match="autre configuration"):
        tracker.start(other)


def test_a_code_change_after_the_freeze_is_flagged(tmp_path, frozen):
    tracker = Tracker(tmp_path / "suivi")
    tracker.start(frozen, today=dt.date(2020, 6, 26))

    def edited(**params):
        return builder.composite(**params)
    edited.__code_fingerprint__ = "something-else"
    out = Tracker(tmp_path / "suivi", rules={"composite": edited}).update(frozen["name"], _prices())
    assert out["statut"] == "CODE_MODIFIÉ"


def test_every_update_stays_on_record(tmp_path, frozen):
    tracker = Tracker(tmp_path / "suivi")
    tracker.start(frozen, today=dt.date(2020, 6, 26))
    tracker.update(frozen["name"], _prices(end="2020-09-30"))
    tracker.update(frozen["name"], _prices(end="2020-12-31"))
    hist = tracker.history(frozen["name"])
    assert list(hist["derniers prix"]) == ["2020-09-30", "2020-12-31"]
    assert tracker.tracked() == [frozen["name"]]
