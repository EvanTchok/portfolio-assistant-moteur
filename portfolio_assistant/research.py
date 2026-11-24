"""The research lab: build on the past, freeze, test once on unseen data.

    lab = Lab(prices, oos_start="2023-01-01")
    lab.explore(spec)        # as many times as you like, on data BEFORE oos_start only
    lab.freeze(spec)         # the strategy is sealed: name, rule, parameters, code
    lab.test_oos("name")     # run once on oos_start -> today, judged against 1/N

    Lab(prices, "2023-01-01", profile="prudent")   # judged for a client profile instead:
    # against that profile's reference portfolio, with its risk aversion and limits
    lab.status()             # everything that was tried, frozen and tested

Why each step exists:

* ``explore`` physically cannot see the test period: it is handed a price panel that
  stops the day before ``oos_start``.
* Every distinct configuration explored is a TRIAL and is counted. Picking the best of
  50 trials is not the same evidence as one idea that worked, and the count says so.
* ``freeze`` seals the configuration. A frozen name cannot be changed; a change is a
  new strategy with a new name, and the log shows both.
* ``test_oos`` is recorded. The first test of a frozen strategy on a holdout nobody has
  consulted yet is the only one marked VALID_OOS. Re-running it, testing after the code
  changed, or testing on a holdout already consulted by an earlier strategy are all
  reported for what they are.
* The log is an append-only JSONL file in YOUR project folder (``research/``), never
  inside the installed package, and there is no function to erase it.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from .compare import compare
from .data import validate_prices
from .engine import BacktestConfig, run_backtest
from .builder import apply_settings, composite
from .strategies import RULES, equal_weight
from . import profiles as _profiles

_PACKAGE_DIR = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------- spec
@dataclass
class StrategySpec:
    """Everything that defines a strategy, as plain data that can be saved."""

    name: str
    rule: str
    params: dict = field(default_factory=dict)
    universe: tuple = ()          # empty = every ticker of the lab
    rebalance: str = "monthly"
    lookback: int = 252
    fee_rate: float = 0.001
    cash_rate: float | str = 0.0  # a fixed rate, or a money-market fund's ticker (see CASH_FUND)
    description: str = ""

    def __post_init__(self):
        self.name = str(self.name).strip()
        if not self.name:
            raise ValueError("a strategy needs a name")
        self.universe = tuple(str(t).strip().upper() for t in self.universe)
        json.dumps(self.params)  # parameters must be saveable as text

    def config(self) -> BacktestConfig:
        # cash_rate is a number (a fixed annual rate; 0 for every strategy frozen so far) or
        # the ticker of a money-market fund whose returns the cash earns
        if isinstance(self.cash_rate, str):
            return BacktestConfig(rebalance=self.rebalance, lookback=self.lookback, fee_rate=self.fee_rate,
                                  cash_prices=cash_prices(self.cash_rate))
        return BacktestConfig(rebalance=self.rebalance, lookback=self.lookback,
                              fee_rate=self.fee_rate, cash_rate=self.cash_rate)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["universe"] = list(self.universe)
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "StrategySpec":
        return cls(**{**data, "universe": tuple(data.get("universe", ()))})


# Money-market fund used for remunerated cash: Xtrackers II EUR Overnight Rate Swap UCITS ETF 1C
# (Xetra, since 25/05/2007, TER 0.10 %), which tracks €STR + 8.5 bp, i.e. the former EONIA.
CASH_FUND = "XEON.DE"
_CASH_CACHE: dict[str, pd.Series] = {}


def _default_cash_loader(ticker: str) -> pd.Series:
    from .market_data import fetch_ticker
    return fetch_ticker(ticker, "2005-01-01")


CASH_LOADER: Callable[[str], pd.Series] = _default_cash_loader


def cash_prices(ticker: str) -> pd.Series:
    """Prices of the money-market fund, downloaded once a day."""
    key = f"{ticker}|{pd.Timestamp.today().date()}"         # refreshed once a day (live tracking)
    if key not in _CASH_CACHE:
        _CASH_CACHE[key] = CASH_LOADER(ticker)
    return _CASH_CACHE[key]


def _source_hash(factory: Callable) -> str:
    """Hash of the rule's code, so an edited rule is detected even under the same name."""
    declared = getattr(factory, "__code_fingerprint__", None)
    if declared:
        return str(declared)
    try:
        text = inspect.getsource(factory)
    except (OSError, TypeError):
        code = getattr(factory, "__code__", None)
        text = repr((code.co_code, code.co_consts)) if code else repr(factory)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------------------ lab
class Lab:
    def __init__(self, prices: pd.DataFrame, oos_start, *, path: str | Path = "research",
                 rules: dict[str, Callable] | None = None, min_years_each_side: float = 1.0,
                 profile: str | None = None, prior_consultations: int = 0):
        self.prices = validate_prices(prices)
        # looks at the SAME test period through other data (e.g. the same indices through
        # other funds): they count like looks at this journal's test period
        self.prior_consultations = int(prior_consultations)
        if self.prices.isna().any().any():
            raise ValueError("the lab needs a full price panel (use fetch_prices or common_window)")
        self.oos_start = pd.Timestamp(oos_start)
        index = self.prices.index
        is_years = (self.oos_start - index[0]).days / 365.25
        oos_years = (index[-1] - self.oos_start).days / 365.25
        if is_years < min_years_each_side or oos_years < min_years_each_side:
            raise ValueError(
                f"oos_start {self.oos_start.date()} leaves {is_years:.1f} years to build and "
                f"{oos_years:.1f} years to test; each side needs at least {min_years_each_side:g}")
        self.is_prices = self.prices.loc[index < self.oos_start]

        self.path = Path(path).expanduser().resolve()
        if self.path == _PACKAGE_DIR or _PACKAGE_DIR in self.path.parents:
            raise ValueError("the research log must live in your project, not inside the package")
        self.path.mkdir(parents=True, exist_ok=True)
        self.log_file = self.path / "registry.jsonl"

        # who judges: 1/N and a γ=3 investor by default, or a client profile. The data set
        # identity does NOT depend on the profile: the test period is shared, so a look at
        # it under one profile counts for all of them.
        self.profile = _profiles.get(profile) if profile else None
        self.profile_name = self.profile.name if self.profile else None
        self.gamma = self.profile.gamma if self.profile else 3.0
        self.rules = dict(RULES)
        self.rules["composite"] = composite
        self.rules.update(rules or {})
        identity = json.dumps({"tickers": sorted(self.prices.columns),
                               "first": str(index[0].date()),
                               "oos_start": str(self.oos_start.date())})
        self.dataset = hashlib.sha256(identity.encode()).hexdigest()[:12]

    # ------------------------------------------------------------------ the log
    def _events(self, kind: str | None = None, *, dataset_only: bool = True) -> list[dict]:
        if not self.log_file.exists():
            return []
        events = [json.loads(line) for line in self.log_file.read_text(encoding="utf-8").splitlines()
                  if line.strip()]
        if dataset_only:
            events = [e for e in events if e.get("dataset") == self.dataset]
        return [e for e in events if kind is None or e["event"] == kind]

    def _append(self, event: dict) -> None:
        event = {"event": event.pop("event"), "at": _now(), "dataset": self.dataset, **event}
        if self.profile_name:
            event["profile"] = self.profile_name
        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, default=str, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ helpers
    def register_rule(self, name: str, factory: Callable) -> None:
        """Adds your own weighting rule. Its code is fingerprinted like the built-in ones."""
        self.rules[name] = factory

    def fingerprint(self, spec: StrategySpec) -> str:
        """Identity of WHAT the strategy does; the name and description are not part of it."""
        if spec.rule not in self.rules:
            raise ValueError(f"unknown rule {spec.rule!r}; available: {sorted(self.rules)}")
        content = spec.to_dict()
        content.pop("name"), content.pop("description")
        content["universe"] = sorted(spec.universe or self.prices.columns)
        content["rule_code"] = _source_hash(self.rules[spec.rule])
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:12]

    def _panel(self, spec: StrategySpec, prices: pd.DataFrame) -> pd.DataFrame:
        universe = list(spec.universe) or list(prices.columns)
        missing = sorted(set(universe) - set(prices.columns))
        if missing:
            raise ValueError(f"tickers not in the lab's prices: {missing}")
        return prices[universe]

    def reference(self, spec: StrategySpec):
        """What ``spec`` is judged against (a robustness.Reference)."""
        from .robustness import Reference

        if not self.profile:
            return Reference(equal_weight, 3.0, "1/N")
        definition = spec.params.get("definition") if spec.rule == "composite" else None
        if not definition:
            raise ValueError("a client profile needs a block strategy (its universe gives the "
                             "asset classes of the reference portfolio)")
        from .builder import _universe
        rule = _profiles.reference_rule(self.profile, _universe(definition))
        return Reference(rule, self.profile.gamma, f"référence {self.profile.label}")

    def _run(self, spec: StrategySpec, prices: pd.DataFrame, start=None) -> dict:
        panel = self._panel(spec, prices)
        rule = self.rules[spec.rule](**spec.params)
        config = spec.config()
        ref = self.reference(spec)
        strategy = run_backtest(panel, rule, config, label=spec.name)
        baseline = run_backtest(panel, ref.rule(), config, label=ref.label)
        out = compare(strategy, baseline, start=start, gamma=self.gamma)
        out["results"] = (strategy, baseline)
        out["reference"] = ref.label
        out["suitability"] = (_profiles.suitability(self.profile, out["strategy"]["volatility"],
                                                    out["strategy"]["max_drawdown"])
                              if self.profile else [])
        if out["suitability"] and out["verdict"] != "INSUFFICIENT_SAMPLE":
            out["verdict"] = "UNSUITABLE"
            out["verdict_reason"] = (f"outside the limits of the {self.profile.label} profile: "
                                     + "; ".join(out["suitability"]))
        return out

    @staticmethod
    def _metrics(out: dict) -> dict:
        s = out["strategy"]
        ce = s["certainty_equivalent"]
        return {"cagr": s["cagr"], "volatility": s["volatility"], "sharpe": s["sharpe"],
                "max_drawdown": s["max_drawdown"], "ce3": ce.get(3.0),
                # kept under its historical key: the gap against the lab's reference,
                # judged with the lab's γ (1/N and γ=3 unless a profile is set)
                "ce3_gap_vs_1N": out["ce_gap"], "gamma": max(ce), "reference": out.get("reference", "1/N"),
                "verdict": out["verdict"],
                "decisions": s.get("n_decisions")}

    def _trial_keys(self) -> set:
        """A trial is a configuration judged by a given judge: the same strategy examined
        for two profiles is two looks, and counts twice."""
        return {(e["fingerprint"], e.get("profile")) for e in self._events("trial")}

    @property
    def n_trials(self) -> int:
        return len(self._trial_keys())

    def holdout_consulted(self) -> int:
        return len(self._events("oos")) + self.prior_consultations

    # ------------------------------------------------------------------ explore
    def explore(self, spec: StrategySpec, *, show: bool = True) -> dict:
        """Runs the strategy and 1/N on the build period only. Counted as a trial."""
        fp = self.fingerprint(spec)
        out = self._run(spec, self.is_prices)
        known = {f for f, prof in self._trial_keys() if prof == self.profile_name}
        after_oos = self.holdout_consulted() > 0
        if fp not in known:
            self._append({"event": "trial", "name": spec.name, "fingerprint": fp,
                          "spec": spec.to_dict(), "is": self._metrics(out), "after_oos": after_oos})
        out["fingerprint"], out["new_trial"], out["n_trials"] = fp, fp not in known, self.n_trials
        if show:
            period = f"{self.is_prices.index[0].date()} -> {self.is_prices.index[-1].date()}"
            print(f"\n[EXPLORATION] {spec.name}  ({period}, données de construction uniquement)")
            print(out["table"].to_string(index=False))
            print(f"verdict: {out['verdict']} - {out['verdict_reason']}")
            print(f"essais comptés sur ce jeu de données : {out['n_trials']}"
                  + ("" if out["new_trial"] else "  (configuration déjà testée, non recomptée)"))
            if after_oos:
                print("ATTENTION : la période de test a déjà été consultée sur ce jeu de données. "
                      "Ce que tu construis maintenant ne pourra plus être validé dessus proprement.")
        return out

    # ------------------------------------------------------------------- freeze
    def freeze(self, spec: StrategySpec, note: str = "", *, show: bool = True) -> dict:
        """Seals a strategy before its test. A frozen name can never change."""
        fp = self.fingerprint(spec)
        for event in self._events("freeze"):
            if event["name"] == spec.name and event.get("profile") == self.profile_name:
                if event["fingerprint"] == fp:
                    if show:
                        print(f"{spec.name} est déjà gelée depuis {event['at']}.")
                    return event
                raise ValueError(
                    f"'{spec.name}' is already frozen with another configuration. A frozen "
                    "strategy is never edited: give the new version a new name (e.g. "
                    f"'{spec.name} v2'). Both will stay in the log.")
        out = self._run(spec, self.is_prices)
        if (fp, self.profile_name) not in self._trial_keys():
            self._append({"event": "trial", "name": spec.name, "fingerprint": fp,
                          "spec": spec.to_dict(), "is": self._metrics(out),
                          "after_oos": self.holdout_consulted() > 0})
        record = {"event": "freeze", "name": spec.name, "fingerprint": fp, "spec": spec.to_dict(),
                  "rule_code": _source_hash(self.rules[spec.rule]), "note": note,
                  "is": self._metrics(out), "n_trials": self.n_trials,
                  "holdout_consulted_before": self.holdout_consulted()}
        self._append(dict(record))
        if show:
            print(f"\n[GEL] {spec.name} gelée (empreinte {fp}). Essais avant le gel : {record['n_trials']}.")
            if record["holdout_consulted_before"]:
                print(f"ATTENTION : la période de test a déjà été consultée "
                      f"{record['holdout_consulted_before']} fois sur ce jeu de données.")
        return record

    # ------------------------------------------------------------------ test
    def test_oos(self, name: str, *, show: bool = True) -> dict:
        """Runs a frozen strategy on the test period, once, and records it."""
        freezes = [e for e in self._events("freeze")
                   if e["name"] == name and e.get("profile") == self.profile_name]
        if not freezes:
            others = sorted({e.get("profile") or "sans profil" for e in self._events("freeze") if e["name"] == name})
            hint = f" (frozen for: {', '.join(others)})" if others else ""
            raise ValueError(f"'{name}' is not frozen for this profile{hint}. "
                             "Freeze it before looking at the test period.")
        frozen = freezes[-1]
        spec = StrategySpec.from_dict(frozen["spec"])
        previous_tests = [e for e in self._events("oos")
                          if e["name"] == name and e.get("profile") == self.profile_name]

        if self.fingerprint(spec) != frozen["fingerprint"]:
            status = "CONTAMINATED_CODE_CHANGED"
            reason = "the rule's code changed after the freeze: this is not the strategy that was sealed"
        elif previous_tests:
            status = "REPEAT"
            reason = f"already tested {len(previous_tests)} time(s): the first result is the one that counts"
        elif frozen["holdout_consulted_before"] > 0:
            status = "HOLDOUT_ALREADY_CONSULTED"
            reason = (f"the test period had already been consulted {frozen['holdout_consulted_before']} "
                      "time(s) when this strategy was frozen: it is a weaker confirmation")
        else:
            status = "VALID_OOS"
            reason = "first test of a strategy frozen before anyone looked at this period"

        out = self._run(spec, self.prices, start=self.oos_start)
        self._append({"event": "oos", "name": name, "fingerprint": frozen["fingerprint"],
                      "status": status, "oos": self._metrics(out)})
        out.update(status=status, status_reason=reason, frozen=frozen)
        if show:
            print(f"\n[TEST HORS ÉCHANTILLON] {name}  "
                  f"({out['start'].date()} -> {self.prices.index[-1].date()})")
            print(out["table"].to_string(index=False))
            print(f"verdict: {out['verdict']} - {out['verdict_reason']}")
            print(f"statut du test : {status} - {reason}")
            print(f"cette stratégie a été retenue parmi {frozen['n_trials']} essai(s).")
            is_cagr, oos_cagr = frozen["is"]["cagr"], out["strategy"]["cagr"]
            print(f"rendement annuel : {is_cagr:.2%} en construction -> {oos_cagr:.2%} en test")
        return out

    # ------------------------------------------------------------------ analyse
    def analyze(self, spec: StrategySpec, *, period: str = "construction",
                grid: dict[str, list] | None = None, show: bool = True) -> dict:
        """Every calculation on one strategy: performance, risk, statistics, robustness.

        ``period="construction"`` works on the build period only. The parameter grid is
        a SEARCH, so every new combination is logged as a trial.
        ``period="test"`` is only allowed once the strategy has been frozen AND tested:
        it then details the test result. Robustness on the test period is limited to
        what cannot be used to tune the strategy (no parameter grid).
        """
        from .analytics import full_analysis
        from .robustness import full_robustness

        if period not in {"construction", "test"}:
            raise ValueError("period must be 'construction' or 'test'")
        if period == "test":
            frozen = [e for e in self._events("freeze")
                      if e["name"] == spec.name and e.get("profile") == self.profile_name]
            tested = [e for e in self._events("oos")
                      if e["name"] == spec.name and e.get("profile") == self.profile_name]
            if not frozen or not tested:
                raise ValueError(
                    f"'{spec.name}' must be frozen and tested with test_oos() before its test "
                    "period can be analysed: looking first would spoil the test.")
            if self.fingerprint(spec) != frozen[-1]["fingerprint"]:
                raise ValueError("this is not the configuration that was frozen under that name")
            if grid:
                raise ValueError("no parameter grid on the test period: that would be tuning on it")
            prices, start = self.prices, self.oos_start
        else:
            self.explore(spec, show=False)
            prices, start = self.is_prices, None

        panel = self._panel(spec, prices)
        factory = self.rules[spec.rule]
        config = spec.config()
        ref = self.reference(spec)
        strategy = run_backtest(panel, factory(**spec.params), config, label=spec.name)
        baseline = run_backtest(panel, ref.rule(), config, label=ref.label)

        if grid and period == "construction":
            known = {f for f, prof in self._trial_keys() if prof == self.profile_name}
            names = list(grid)
            import itertools
            for combo in itertools.product(*(grid[n] for n in names)):
                variant = StrategySpec.from_dict({**spec.to_dict(),
                                                  "name": f"{spec.name} {dict(zip(names, combo))}",
                                                  "params": apply_settings(spec.params, dict(zip(names, combo)))})
                fp = self.fingerprint(variant)
                if fp not in known:
                    out = self._run(variant, self.is_prices)
                    self._append({"event": "trial", "name": variant.name, "fingerprint": fp,
                                  "spec": variant.to_dict(), "is": self._metrics(out),
                                  "after_oos": self.holdout_consulted() > 0, "source": "grid"})
                    known.add(fp)

        trial_sharpes = [e["is"]["sharpe"] for e in self._events("trial")
                         if e["is"].get("sharpe") is not None]
        analysis = full_analysis(strategy, baseline, start=start, n_trials=max(self.n_trials, 1),
                                 trial_sharpes=trial_sharpes if len(trial_sharpes) > 1 else None)
        if period == "test":
            from .robustness import block_bootstrap, rolling_windows, rolling_summary
            nav, base = analysis["courbes"][spec.name], analysis["courbes"][ref.label]
            robustness = {"bootstrap": block_bootstrap(nav, base, ref=ref)}
            windows = rolling_windows(nav, base, years=1, ref=ref)
            robustness["fenêtres 1 an"] = windows
            robustness["fenêtres 1 an - synthèse"] = pd.Series(rolling_summary(windows), name="valeur").to_frame()
        else:
            robustness = full_robustness(panel, factory, spec.params, config, grid=grid, ref=ref)
        report = {"spec": spec, "période": period, "analyse": analysis, "robustesse": robustness,
                  "essais": self.n_trials, "jeu de données": self.dataset,
                  "fenêtre": (str(prices.index[0].date()), str(prices.index[-1].date())),
                  "oos_start": str(self.oos_start.date()),
                  "profil": self.profile.to_dict() if self.profile else None}
        if show:
            a = analysis
            print(f"\n[ANALYSE {period.upper()}] {spec.name}  {a['période']['début']} -> {a['période']['fin']}")
            for block in ("rendement", "risque", "ratios"):
                print(f"\n-- {block}")
                print(a[block].to_string())
            print("\n-- face à 1/N"); print(a["relatif"].to_string())
            print("\n-- chance ou talent ?"); print(a["statistiques"].to_string())
        return report

    # ------------------------------------------------------------------ reading
    def trials(self) -> pd.DataFrame:
        rows = [{"date": e["at"], "nom": e["name"], "empreinte": e["fingerprint"],
                 "règle": e["spec"]["rule"], "paramètres": json.dumps(e["spec"]["params"]),
                 "CAGR": e["is"]["cagr"], "Sharpe": e["is"]["sharpe"],
                 "pire chute": e["is"]["max_drawdown"], "écart CE vs 1/N": e["is"]["ce3_gap_vs_1N"],
                 "après test": e.get("after_oos", False)} for e in self._events("trial")]
        return pd.DataFrame(rows)

    def status(self) -> pd.DataFrame:
        rows = []
        tests = self._events("oos")
        for freeze in self._events("freeze"):
            mine = [t for t in tests if t["name"] == freeze["name"]]
            first = mine[0] if mine else None
            rows.append({
                "stratégie": freeze["name"], "gelée le": freeze["at"][:10],
                "essais avant gel": freeze["n_trials"],
                "CAGR construction": freeze["is"]["cagr"],
                "CAGR test": first["oos"]["cagr"] if first else None,
                "verdict test": first["oos"]["verdict"] if first else "pas encore testée",
                "statut": first["status"] if first else "-",
                "tests": len(mine),
            })
        return pd.DataFrame(rows)
