"""The research lab: the test period stays sealed, trials are counted, tests recorded."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import portfolio_assistant as pa
from portfolio_assistant import Lab, StrategySpec

OOS = "2023-01-02"


def _prices(seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", "2026-09-30")
    rets = rng.normal(0.0003, 0.01, (len(idx), 4))
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=list("ABCD"))


def _lab(tmp_path, **kw):
    return Lab(_prices(), OOS, path=tmp_path / "research", **kw)


def _momo(name="momo", k=2):
    return StrategySpec(name=name, rule="momentum_top_k", params={"lookback": 252, "k": k})


def test_exploration_never_sees_the_test_period(tmp_path):
    seen = []

    def spy_rule():
        def fn(ctx):
            seen.append(ctx.history.index[-1])
            return {"A": 1.0}
        return fn

    lab = _lab(tmp_path, rules={"spy": spy_rule})
    out = lab.explore(StrategySpec(name="spy", rule="spy", lookback=0), show=False)
    assert seen and max(seen) < pd.Timestamp(OOS)
    strategy, _ = out["results"]
    assert strategy.nav.index[-1] < pd.Timestamp(OOS)


def test_a_configuration_is_counted_once_and_a_new_one_adds_one(tmp_path):
    lab = _lab(tmp_path)
    lab.explore(_momo(k=2), show=False)
    lab.explore(_momo(k=2), show=False)
    assert lab.n_trials == 1
    lab.explore(_momo(name="renamed", k=2), show=False)      # same strategy, another name
    assert lab.n_trials == 1
    lab.explore(_momo(k=3), show=False)
    assert lab.n_trials == 2


def test_a_changed_rule_code_is_a_new_trial_even_with_the_same_name(tmp_path):
    def v1():
        return lambda ctx: {"A": 1.0}

    def v2():
        return lambda ctx: {"B": 1.0}

    lab = _lab(tmp_path, rules={"mine": v1})
    spec = StrategySpec(name="mine", rule="mine", lookback=0)
    first = lab.fingerprint(spec)
    lab.register_rule("mine", v2)
    assert lab.fingerprint(spec) != first


def test_testing_requires_a_freeze(tmp_path):
    lab = _lab(tmp_path)
    lab.explore(_momo(), show=False)
    with pytest.raises(ValueError, match="not frozen"):
        lab.test_oos("momo", show=False)


def test_first_test_is_valid_and_a_second_is_a_repeat(tmp_path):
    lab = _lab(tmp_path)
    lab.freeze(_momo(), show=False)
    first = lab.test_oos("momo", show=False)
    second = lab.test_oos("momo", show=False)
    assert first["status"] == "VALID_OOS" and second["status"] == "REPEAT"
    assert first["start"] >= pd.Timestamp(OOS)
    assert first["strategy"]["cagr"] == second["strategy"]["cagr"]


def test_a_frozen_name_cannot_be_edited(tmp_path):
    lab = _lab(tmp_path)
    lab.freeze(_momo(k=2), show=False)
    lab.freeze(_momo(k=2), show=False)                       # same thing: accepted, no-op
    with pytest.raises(ValueError, match="new name"):
        lab.freeze(_momo(k=3), show=False)


def test_a_strategy_frozen_after_the_holdout_was_seen_is_flagged(tmp_path):
    lab = _lab(tmp_path)
    lab.freeze(_momo(), show=False)
    lab.test_oos("momo", show=False)
    out = lab.explore(_momo(name="momo v2", k=3), show=False)
    assert lab.trials()["après test"].iloc[-1]
    lab.freeze(_momo(name="momo v2", k=3), show=False)
    assert lab.test_oos("momo v2", show=False)["status"] == "HOLDOUT_ALREADY_CONSULTED"
    assert out["n_trials"] == 2


def test_code_changed_after_freeze_is_contaminated(tmp_path):
    def v1():
        return lambda ctx: {"A": 1.0}

    def v2():
        return lambda ctx: {"B": 1.0}

    lab = _lab(tmp_path, rules={"mine": v1})
    lab.freeze(StrategySpec(name="mine", rule="mine", lookback=0), show=False)
    lab.register_rule("mine", v2)
    assert lab.test_oos("mine", show=False)["status"] == "CONTAMINATED_CODE_CHANGED"


def test_the_log_survives_a_restart_and_is_per_dataset(tmp_path):
    lab = _lab(tmp_path)
    lab.freeze(_momo(), show=False)
    lab.test_oos("momo", show=False)
    again = _lab(tmp_path)
    assert again.n_trials == 1 and again.holdout_consulted() == 1
    other = Lab(_prices(), "2022-01-03", path=tmp_path / "research")   # another split
    assert other.n_trials == 0 and other.holdout_consulted() == 0
    status = again.status()
    assert status.loc[0, "statut"] == "VALID_OOS" and status.loc[0, "tests"] == 1


def test_the_log_cannot_live_inside_the_package():
    package_dir = pa.__file__.rsplit("/", 1)[0]
    with pytest.raises(ValueError, match="not inside the package"):
        Lab(_prices(), OOS, path=package_dir + "/research")


def test_split_must_leave_time_on_both_sides(tmp_path):
    with pytest.raises(ValueError, match="each side"):
        Lab(_prices(), "2026-06-01", path=tmp_path / "r")
