"""What the web interface does, as plain functions (testable without a browser).

Every action goes through the same lab, journals and tracker as the scripts: a strategy
explored from the interface is a counted trial, a freeze is a freeze, a test is the one
and only test. The interface adds convenience, never a back door.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable

import pandas as pd

from . import builder, chemins, profiles, universes
from .research import Lab
from .synthesis import collect
from .tracking import Tracker, frozen_from_journal

OOS_START = "2023-01-01"


# ----------------------------------------------------------------------------- context
class Workspace:
    """The project folder: journals, saved strategies, tracking, reports, price cache."""

    spend: Callable | None = None              # credits taken before a calculation (personal labs online)

    def __init__(self, root: str | Path, price_loader: Callable | None = None):
        self.root = Path(root)
        self.lab_dir = chemins.lab(self.root)               # registries, strategies, tracking, reports
        self.data = chemins.cabinet(self.root)              # this installation's own data (see chemins.py)
        universes.use_folder(self.data / "univers")        # the adviser's own universes
        self.cache_dir = chemins.cache(self.root)
        self._loader = price_loader or universes.load_prices
        self._prices: dict[str, tuple] = {}          # universe -> (day loaded, panel); shared by the cabinets

    # prices are downloaded once per day and universe (a server stays up for weeks: a panel
    # loaded at start would otherwise never see a new closing price)
    def prices(self, universe: str) -> pd.DataFrame:
        from . import sources_prix
        today = pd.Timestamp.today().date()
        held = self._prices.get(universe)
        if held is None or held[0] != today:
            with sources_prix.sans_cours_importes():      # shared panel: never a cabinet's own imports
                try:
                    panel = self._loader(universe, cache_dir=self.cache_dir, verbose=False)
                except Exception:
                    if held is None:
                        raise
                    panel = held[1]                        # the source is down today: yesterday's panel stays
            self._prices[universe] = held = (today, panel)
        return held[1]

    def lab(self, universe: str, profile: str | None) -> Lab:
        from . import code_rules
        return Lab(self.prices(universe), OOS_START, path=self.root / universes.get(universe)["journal"],
                   prior_consultations=universes.prior_consultations(universe, self.root),
                   profile=profile or None, rules=code_rules.load_all(self.lab_dir))

    def status(self) -> dict:
        return collect(self.root)

    def for_cabinet(self, data: str | Path) -> "Workspace":
        """The same lab and prices, with another cabinet's own folder (the online version gives
        each adviser his own: see comptes.py)."""
        ws = copy.copy(self)                   # shares the downloaded prices
        ws.data = Path(data)
        return ws

    def for_account(self, folder: str | Path, spend: Callable | None = None) -> "Workspace":
        """A personal lab (online traders, the demo): the same prices, but its own registries,
        trials, frozen strategies, tracking and workshop in ``folder``; ``spend(credits, what)``
        is called before each calculation and may refuse it."""
        ws = copy.copy(self)
        ws.root = Path(folder)
        ws.data = Path(folder)
        ws.lab_dir = chemins.lab(folder)
        ws.spend = spend
        return ws

    # ------------------------------------------------------------- saved strategies
    @property
    def strategies_dir(self) -> Path:
        return self.lab_dir / "strategies"

    def saved(self) -> dict[str, dict]:
        out = {}
        if self.strategies_dir.exists():
            for path in sorted(self.strategies_dir.glob("*.json")):
                try:
                    d = builder.load_definition(path)
                    out[d.get("nom", path.stem)] = d
                except Exception:          # a hand-edited, broken file must not break the page
                    continue
        return out

    def save(self, definition: dict) -> Path:
        return builder.save_definition(definition, self.strategies_dir)


def _spend(ws: Workspace, credits: int, what: str) -> None:
    if getattr(ws, "spend", None):
        ws.spend(int(credits), what)


COSTS = {"explorer": 1, "robustesse": 3, "tester": 1, "valider": 5, "atelier": 1, "algo": 1, "recherche": 1}


# ------------------------------------------------------------------------- building
def new_definition(universe: str) -> dict:
    """An empty strategy on a universe: every asset, equal weights, monthly, 0.10 % fees."""
    return {"nom": "Ma stratégie", "univers": universes.assets(universe),
            "ponderation": {"type": "equal"}, "rebalancement": "monthly", "frais": 0.001}


def clean(definition: dict, like: dict | None = None) -> dict:
    """Drops empty optional settings so that two identical strategies get identical files
    (and identical fingerprints in the journal)."""
    d = copy.deepcopy(definition)
    d["univers"] = builder._universe(d)          # same spelling as the ready-made strategies
    d["contraintes"] = {k: v for k, v in d.get("contraintes", {}).items()
                        if v not in (None, {}, []) and not (k == "max_weight" and v >= 1)
                        and not (k in ("min_weight", "min_cash") and v == 0)}
    d["risque"] = {k: v for k, v in d.get("risque", {}).items() if v not in (None, "")}
    if d.get("liquidités") != "monétaire":           # unpaid cash is the default: not written
        d.pop("liquidités", None)
    if "target_vol" not in d["risque"]:
        d["risque"].pop("days", None)
    # empty parts are left out, exactly as the ready-made strategies write them: the same
    # strategy loaded in the interface must keep the SAME fingerprint, so that exploring it
    # again is not counted as a new trial
    # a setting left at its default is dropped when the loaded block did not write it
    # (the form shows every setting; the ready-made strategies only write the ones they set)
    if like is not None:
        def trim(block: dict, ref: dict | None, catalog) -> dict:
            if not ref or ref.get("type") != block.get("type"):
                return block
            defaults = {p.name: p.default for p in catalog[block["type"]].params}
            return {k: v for k, v in block.items()
                    if k == "type" or k in ref or v != defaults.get(k, object())}
        ref_sel = like.get("selection", [])
        d["selection"] = [trim(b, ref_sel[i] if i < len(ref_sel) else None, builder.SELECTIONS)
                          for i, b in enumerate(d.get("selection", []))]
        if "ponderation" in d:
            d["ponderation"] = trim(d["ponderation"], like.get("ponderation"), builder.WEIGHTINGS)
    # (``like`` is the strategy that was loaded: an empty part it had is kept, one it did
    # not have is dropped; by default the definition keeps its own shape)
    shape = like if like is not None else definition
    for key in ("selection", "contraintes", "risque", "description"):
        if key in d and not d[key] and key not in shape:
            del d[key]
    return d


def to_spec(definition: dict):
    """The lab spec of a definition; « liquidités : monétaire » makes the cash earn the
    money-market fund instead of 0 %."""
    import dataclasses
    from .research import CASH_FUND
    spec = builder.to_spec(definition)
    if definition.get("liquidités") == "monétaire":
        spec = dataclasses.replace(spec, cash_rate=CASH_FUND)
    return spec


def auto_grid(definition: dict) -> dict[str, list]:
    """Neighbouring settings for the robustness test: the main look-back of the strategy
    at ×0.5, ×1 and ×1.5 (the first one found, selection first, then weighting)."""
    for i, block in enumerate(definition.get("selection", [])):
        if "days" in block or any(p.name == "days" for p in builder.SELECTIONS[block["type"]].params):
            base = int(block.get("days", next(p.default for p in builder.SELECTIONS[block["type"]].params
                                                if p.name == "days")))
            return {f"selection.{i}.days": sorted({max(20, round(base * f)) for f in (0.5, 1, 1.5)})}
    w = definition.get("ponderation", {})
    params = builder.WEIGHTINGS[w.get("type", "equal")].params
    if any(p.name == "days" for p in params):
        base = int(w.get("days", next(p.default for p in params if p.name == "days")))
        return {"ponderation.days": sorted({max(20, round(base * f)) for f in (0.5, 1, 1.5)})}
    return {}


# --------------------------------------------------------------------------- the lab
def _curves(out: dict) -> pd.DataFrame:
    strategy, baseline = out["results"]
    navs = pd.DataFrame({strategy.label: strategy.nav, baseline.label: baseline.nav}).loc[out["start"]:]
    return navs / navs.iloc[0] * 100


def _metrics(out: dict) -> pd.DataFrame:
    s, b = out["strategy"], out["baseline"]
    rows = [("Rendement annuel", s["cagr"], b["cagr"]), ("Volatilité", s["volatility"], b["volatility"]),
            ("Sharpe", s["sharpe"], b["sharpe"]), ("Pire chute", s["max_drawdown"], b["max_drawdown"])]
    return pd.DataFrame(rows, columns=["", "Stratégie", out.get("reference", "Référence")]).set_index("")


def explore(ws: Workspace, definition: dict, universe: str, profile: str | None) -> dict:
    """Build period only. Counted as a trial (once per configuration and judge)."""
    _spend(ws, COSTS['explorer'], 'explorer')
    definition = clean(definition)
    errors = builder.validate(definition, available=list(ws.prices(universe).columns))
    if errors:
        return {"errors": errors}
    lab = ws.lab(universe, profile)
    out = lab.explore(to_spec(definition), show=False)
    return {"errors": [], "verdict": out["verdict"], "reason": out["verdict_reason"], "gap": out["ce_gap"],
            "suitability": out.get("suitability", []), "metrics": _metrics(out), "curves": _curves(out),
            "new_trial": out["new_trial"], "n_trials": out["n_trials"],
            "holdout_consulted": lab.holdout_consulted(), "start": out["start"],
            "end": lab.is_prices.index[-1]}


ROBUSTNESS_THRESHOLDS = {"réglages": 0.70, "frais": 0.005, "fenêtres": 0.60, "bootstrap": 0.60, "chance": 0.90}


def robustness(ws: Workspace, definition: dict, universe: str, profile: str | None) -> dict:
    """The five pre-registered tests and the composition, on the build period."""
    _spend(ws, COSTS['robustesse'], 'robustesse')
    definition = clean(definition)
    lab = ws.lab(universe, profile)
    grid = auto_grid(definition)
    report = lab.analyze(to_spec(definition), grid=grid or None, show=False)
    rob, stats = report["robustesse"], report["analyse"]["statistiques"]

    def value(frame, starts):
        rows = [k for k in frame.index if str(k).startswith(starts)]
        return float(frame.loc[rows[0], "valeur"]) if rows else float("nan")

    rolling = next(v for k, v in rob.items() if k.startswith("fenêtres") and k.endswith("synthèse"))
    boot = rob["bootstrap"].set_index("mesure")
    scores = {
        "réglages": value(rob["paramètres - synthèse"], "% des combinaisons") if grid else float("nan"),
        "frais": float(rob["frais - seuil de rentabilité"]),
        "fenêtres": float(rolling.loc[[k for k in rolling.index if "ajusté" in k][0], "valeur"]),
        "bootstrap": float(boot.loc[[m for m in boot.index if m.startswith("écart CE")][0], "probabilité > 0"]),
        "chance": float(stats.loc[[k for k in stats.index if "DSR" in k][0], "valeur"]),
    }
    passed = {k: (v >= ROBUSTNESS_THRESHOLDS[k]) if v == v else None for k, v in scores.items()}
    comp = report["analyse"]["composition"]
    share_col = next((c for c in comp.columns if "part du rendement" in c), None)
    top = (comp[share_col].astype(float).idxmax(), float(comp[share_col].astype(float).max())) if share_col else None
    return {"scores": scores, "passed": passed, "grid": grid, "composition": comp, "top_asset": top,
            "n_passed": sum(bool(p) for p in passed.values()), "n_tests": sum(p is not None for p in passed.values()),
            "report": report, "n_trials": lab.n_trials}


def freeze(ws: Workspace, definition: dict, universe: str, profile: str | None) -> dict:
    lab = ws.lab(universe, profile)
    return lab.freeze(to_spec(clean(definition)), note="gelée depuis l'interface", show=False)


def frozen_list(ws: Workspace) -> pd.DataFrame:
    """Every frozen strategy, with its test if any."""
    rows = []
    for name, j in ws.status()["journals"].items():
        tested = {(t["name"], t["profile"]): t for t in j["tests"]}
        for f in j["freezes"]:
            t = tested.get((f["name"], f.get("profile")))
            rows.append({"univers": name, "profil": f.get("profile") or "", "stratégie": f["name"],
                         "gelée le": f["at"][:10], "essais avant gel": f["n_trials"],
                         "statut": t["status"] if t else "pas encore testée",
                         "verdict": t["oos"]["verdict"] if t else "",
                         "écart": t["oos"].get("ce3_gap_vs_1N") if t else None})
    return pd.DataFrame(rows)


def test(ws: Workspace, name: str, universe: str, profile: str | None) -> dict:
    """THE test on 2023 → today. Recorded; running it twice only gives REPEAT."""
    _spend(ws, COSTS['tester'], 'tester')
    lab = ws.lab(universe, profile)
    out = lab.test_oos(name, show=False)
    return {"status": out["status"], "status_reason": out["status_reason"], "verdict": out["verdict"],
            "reason": out["verdict_reason"], "gap": out["ce_gap"], "metrics": _metrics(out),
            "curves": _curves(out)}


# ----------------------------------------------------------------------- validation
def validate(ws: Workspace, universe: str, profile: str | None, *, name: str | None = None,
             definition: dict | None = None, n_baskets: int = 40, progress=None) -> dict:
    """The twelve validation tests (validation_tests.py) on a frozen strategy (by name) or on a
    strategy being built (its definition). Construction period only, never counted as a trial;
    the report is kept in laboratoire/validations/."""
    _spend(ws, COSTS['valider'], 'valider')
    import json
    from . import validation_tests as vt
    from .research import StrategySpec
    lab = ws.lab(universe, profile)
    if name:
        frozen = [e for e in lab._events("freeze") if e["name"] == name and e.get("profile") == (profile or None)]
        if not frozen:
            raise ValueError(f"« {name} » n'est pas gelée dans cet univers pour ce profil")
        spec = StrategySpec.from_dict(frozen[-1]["spec"])
    else:
        spec = to_spec(definition)
    pool, classes = vt.reservoir_prices(ws.cache_dir, base_currency=universes.get(universe)["devise"])
    verdicts = vt.battery(lab, spec, pool=pool, pool_classes=classes, n_baskets=n_baskets, progress=progress)
    report = {"stratégie": spec.name, "univers": universe, "profil": profile, "empreinte": lab.fingerprint(spec),
              "date": pd.Timestamp.now().isoformat(timespec="seconds"), "essais au registre": lab.n_trials,
              "synthèse": vt.summary(verdicts), "tests": [v.to_dict() for v in verdicts]}
    folder = ws.lab_dir / "validations"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{spec.name}_{universe}_{profile or 'sans-profil'}".replace(" ", "_").replace("/", "-")
    (folder / f"{stem}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return {"report": report, "verdicts": verdicts}


def last_validation(ws: Workspace, name: str, universe: str, profile: str | None) -> dict | None:
    import json
    stem = f"{name}_{universe}_{profile or 'sans-profil'}".replace(" ", "_").replace("/", "-")
    path = ws.lab_dir / "validations" / f"{stem}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, ValueError):
        return None


# -------------------------------------------------------------------------- atelier
ATELIER_RULES = {
    "Momentum long/short (3 meilleurs contre 3 moins bons, 12 mois)": ("long_short_momentum",
                                                                       {"lookback": 252, "k": 3, "skip": 21}),
    "Tendance long/short (au-dessus ou au-dessous de la moyenne 200 jours)": ("trend_long_short", {"days": 200}),
}


def atelier_instruments(universe: str) -> list[dict]:
    """Default kind of each line: the universes hold ETFs (bond ETFs priced like bonds)."""
    rows = []
    for a in universes.assets(universe):
        kind = "forex" if a["ticker"].endswith("=X") else ("obligation" if "oblig" in a["classe"] else "etf")
        rows.append({"ticker": a["ticker"], "nom": a["nom"], "type": kind, "devise": a.get("devise", "EUR"),
                     "TTF": False, "droit de timbre": False})
    return rows


def atelier_choices(ws: Workspace, universe: str, profile: str | None) -> dict[str, dict]:
    """What the workshop can run: frozen strategies, saved ones, presets, long/short rules."""
    lab = ws.lab(universe, profile)
    out = {}
    for e in lab._events("freeze"):
        if e.get("profile") == (profile or None):
            out[f"Gelée — {e['name']}"] = {"kind": "spec", "spec": e["spec"]}
    for name, d in ws.saved().items():
        if d.get("univers", universe) == universe:
            out[f"Enregistrée — {name}"] = {"kind": "definition", "definition": d}
    for name, d in builder.presets(universes.assets(universe)).items():
        out[f"Modèle — {name}"] = {"kind": "definition", "definition": d}
    for label, (rule, params) in ATELIER_RULES.items():
        out[f"Atelier — {label}"] = {"kind": "advanced", "rule": rule, "params": params}
    return out


def _atelier_log(ws: Workspace, identity: dict, record: dict) -> int:
    """Logs a workshop configuration once; returns how many distinct ones were tried."""
    import hashlib
    import json
    key = json.dumps(identity, sort_keys=True, default=str, ensure_ascii=False)
    fp = hashlib.sha256(key.encode()).hexdigest()[:12]
    folder = ws.lab_dir / "atelier"
    folder.mkdir(parents=True, exist_ok=True)
    log = folder / "registre.jsonl"
    seen = set()
    if log.exists():
        seen = {json.loads(line)["empreinte"] for line in log.read_text(encoding="utf-8").splitlines() if line.strip()}
    if fp not in seen:
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"à": pd.Timestamp.now().isoformat(timespec="seconds"), "empreinte": fp, **record},
                                    ensure_ascii=False, default=str) + "\n")
        seen.add(fp)
    return len(seen)


def _atelier_log_many(ws: Workspace, entries: list[tuple[dict, dict]]) -> int:
    """Logs many configurations at once (a grid); returns the workshop's distinct count."""
    import hashlib
    import json
    folder = ws.lab_dir / "atelier"
    folder.mkdir(parents=True, exist_ok=True)
    log = folder / "registre.jsonl"
    seen = set()
    if log.exists():
        seen = {json.loads(line)["empreinte"] for line in log.read_text(encoding="utf-8").splitlines() if line.strip()}
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    with log.open("a", encoding="utf-8") as handle:
        for identity, record in entries:
            fp = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()[:12]
            if fp not in seen:
                handle.write(json.dumps({"à": now, "empreinte": fp, **record}, ensure_ascii=False, default=str) + "\n")
                seen.add(fp)
    return len(seen)


def grid_run(ws: Workspace, universe: str, settings: dict, grid: dict[str, list],
             instruments: list[dict] | None = None, progress=None) -> dict:
    """A grid of the algorithmic mode on the construction period; every combination is
    counted in the workshop."""
    from . import algo
    from . import execution as ex
    from . import grille

    lab = ws.lab(universe, None)
    prices = lab.is_prices
    fees = settings.get("frais", next(iter(ex.COST_PRESETS)))
    base = algo.AlgoSpec(costs=ex.COST_PRESETS[fees], **{k: v for k, v in settings.items() if k not in ("frais", "en séance")})
    kinds = {r["ticker"]: ex.Instrument(r["ticker"], r["type"], ttf=bool(r.get("TTF")), stamp=bool(r.get("droit de timbre")),
                                        currency=str(r.get("devise") or "EUR"))
             for r in (instruments or atelier_instruments(universe)) if r["ticker"] in prices.columns}
    _spend(ws, max(1, -(-len(grille.combinations(base, grid)) // 10)), "grille")
    out = grille.run_grid(prices, base, grid, kinds, progress=progress)
    entries = [({"mode": "algo", "univers": universe, "réglages": {**settings, **params}, "instruments": instruments,
                 "jeu": lab.dataset},
                {"stratégie": f"algo {base.signal} (grille)", "univers": universe, "réglages": {**settings, **params}})
               for params, _ in grille.combinations(base, grid)]
    out["n_configurations"] = _atelier_log_many(ws, entries)
    out["period"] = (str(prices.index[base.warmup].date()), str(prices.index[-1].date()))
    return out


def research_study(ws: Workspace, universe: str, settings: dict, *, horizons=(5, 21, 63),
                   target: float | None = None, stop: float | None = None) -> dict:
    """Research mode on the construction period: what each occurrence of a signal was
    followed by. Counted in the workshop like any configuration looked at."""
    _spend(ws, COSTS['recherche'], 'recherche')
    from . import algo, recherche
    lab = ws.lab(universe, None)
    spec = algo.AlgoSpec(**{k: v for k, v in settings.items() if k != "frais"})
    out = recherche.study(lab.is_prices, spec, horizons=tuple(horizons), target=target, stop=stop)
    out["n_configurations"] = _atelier_log(
        ws, {"mode": "recherche", "univers": universe, "réglages": settings, "horizons": list(horizons),
             "objectif": target, "stop": stop, "jeu": lab.dataset},
        {"stratégie": f"recherche {spec.signal}", "univers": universe, "réglages": settings})
    out["period"] = (str(lab.is_prices.index[spec.warmup].date()), str(lab.is_prices.index[-1].date()))
    return out


def blind_session(ws: Workspace, universe: str, seed: int):
    """A practice window drawn from the construction period only."""
    from . import recherche
    return recherche.Blind.draw(ws.lab(universe, None).is_prices, seed)


def algo_run(ws: Workspace, universe: str, settings: dict, instruments: list[dict] | None = None) -> dict:
    """The algorithmic mode on the construction period, against 1/N rebalanced monthly."""
    _spend(ws, COSTS['algo'], 'algo')
    from . import algo
    from . import execution as ex
    from .engine import BacktestConfig, run_backtest
    from .metrics import summary
    from .strategies import equal_weight

    lab = ws.lab(universe, None)
    prices = lab.is_prices
    fees = settings.get("frais", next(iter(ex.COST_PRESETS)))
    settings_session = bool(settings.get("en séance"))
    params = {k: v for k, v in settings.items() if k not in ("frais", "en séance")}
    spec = algo.AlgoSpec(costs=ex.COST_PRESETS[fees], **params)
    kinds = {r["ticker"]: ex.Instrument(r["ticker"], r["type"], ttf=bool(r.get("TTF")), stamp=bool(r.get("droit de timbre")),
                                        currency=str(r.get("devise") or "EUR"))
             for r in (instruments or atelier_instruments(universe)) if r["ticker"] in prices.columns}
    ranges, missing = (None, [])
    if settings_session:
        from .market_data import fetch_ranges
        ranges, missing = fetch_ranges(prices, cache_dir=ws.cache_dir)
        if len(missing) == len(prices.columns):
            ranges = None
    res = algo.run_algo(prices, spec, kinds, ranges=ranges)
    start = prices.index[spec.warmup]
    ref = run_backtest(prices.loc[start:], equal_weight(), BacktestConfig(lookback=0)).nav
    navs = {"Système": res.nav.loc[start:], "1/N rééquilibré chaque mois": ref}
    curves = pd.DataFrame(navs)
    curves = curves / curves.iloc[0] * 100
    rows = {}
    for label, nav in navs.items():
        s = summary(nav)
        rows[label] = {"Rendement annuel": s["cagr"], "Volatilité": s["volatility"], "Sharpe": s["sharpe"],
                       "Pire chute": s["max_drawdown"]}
    n = _atelier_log(ws, {"mode": "algo", "univers": universe, "réglages": settings, "instruments": instruments,
                          "jeu": lab.dataset},
                     {"stratégie": f"algo {spec.signal}", "univers": universe, "réglages": settings,
                      "rendement annuel": rows["Système"]["Rendement annuel"]})
    return {"result": res, "metrics": pd.DataFrame(rows), "curves": curves, "stats": res.stats(),
            "costs": ex.cost_table(ex.AdvancedResult(result=_Capital(spec.initial_capital), costs=res.costs,
                                                     events=pd.DataFrame())),
            "n_configurations": n, "period": (str(start.date()), str(prices.index[-1].date())),
            "séance": None if not settings_session else (
                "indisponible : aucun plus haut / plus bas téléchargé (connexion ?), exécution aux clôtures"
                if ranges is None else ("plus hauts et plus bas utilisés" + (f" (sans : {', '.join(missing)})" if missing else "")))}


def code_spec(name: str, rule: str, rebalance: str = "monthly", lookback: int = 0):
    from .research import StrategySpec
    return StrategySpec(name=name, rule=rule, rebalance=rebalance, lookback=int(lookback),
                        description="stratégie écrite en code")


def code_save(ws: Workspace, name: str, source: str) -> str:
    from . import code_rules
    if not code_rules.editor_allowed():
        raise PermissionError("l'écriture de code n'est possible que sur un ordinateur, jamais en ligne")
    return code_rules.save(ws.lab_dir, name, source)


def code_explore(ws: Workspace, universe: str, profile: str | None, name: str, rule: str,
                 rebalance: str = "monthly", lookback: int = 0) -> dict:
    """The saved code in the lab itself: counted as a trial, like any strategy. Judged
    against 1/N: a profile's reference portfolio is built from a block strategy's classes."""
    _spend(ws, COSTS['explorer'], 'explorer')
    if profile:
        raise ValueError("une stratégie en code est jugée contre 1/N, sans profil : la référence d'un profil "
                         "se construit à partir des classes d'actifs d'une stratégie en blocs")
    lab = ws.lab(universe, profile)
    out = lab.explore(code_spec(name, rule, rebalance, lookback), show=False)
    return {"verdict": out["verdict"], "reason": out["verdict_reason"], "gap": out["ce_gap"],
            "metrics": _metrics(out), "curves": _curves(out), "new_trial": out["new_trial"],
            "n_trials": out["n_trials"], "suitability": out.get("suitability", [])}


def code_freeze(ws: Workspace, universe: str, profile: str | None, name: str, rule: str,
                rebalance: str = "monthly", lookback: int = 0) -> dict:
    if profile:
        raise ValueError("une stratégie en code se gèle sans profil (jugée contre 1/N)")
    return ws.lab(universe, profile).freeze(code_spec(name, rule, rebalance, lookback),
                                            note="stratégie en code, gelée depuis l'atelier", show=False)


class _Capital:                       # what cost_table reads from a result
    def __init__(self, capital: float):
        self.config = type("C", (), {"initial_capital": capital})()


def atelier_run(ws: Workspace, universe: str, profile: str | None, choice: dict, settings: dict) -> dict:
    """One run in the workshop, on the construction period only. Logged and counted apart
    from the lab's trials: nothing here can be frozen, and the count says how many
    configurations were tried."""
    _spend(ws, COSTS['atelier'], 'atelier')
    from . import execution as ex
    from .metrics import summary
    from .research import StrategySpec, cash_prices

    lab = ws.lab(universe, profile)
    prices = lab.is_prices
    if choice["kind"] == "code":
        from . import code_rules
        rule = code_rules.compile_rule(choice["source"])
        panel, lookback, rebalance, cash, name = prices, int(choice.get("lookback", 0)), \
            choice.get("rebalance", "monthly"), 0.0, choice.get("name", "code")
    elif choice["kind"] == "advanced":
        rule = getattr(ex, choice["rule"])(**choice["params"])
        panel, lookback, rebalance, cash = prices, int(choice["params"].get("lookback", choice["params"].get("days", 252))) + 5, \
            "monthly", 0.0
        name = choice["rule"]
    else:
        spec = StrategySpec.from_dict(choice["spec"]) if choice["kind"] == "spec" else to_spec(choice["definition"])
        rule = lab.rules[spec.rule](**spec.params)
        panel, lookback, rebalance, cash, name = lab._panel(spec, prices), spec.lookback, spec.rebalance, \
            spec.cash_rate, spec.name
    leverage = float(settings.get("levier", 1.0))
    short = bool(settings.get("vente à découvert", False)) or choice["kind"] == "advanced"
    cfg = dict(initial_capital=float(settings.get("capital", 100_000)), rebalance=rebalance, lookback=lookback,
               costs=ex.COST_PRESETS[settings.get("frais", next(iter(ex.COST_PRESETS)))],
               leverage=leverage, max_gross=max(leverage, 1.0), allow_short=short,
               debit_spread=float(settings.get("marge d'emprunt", 0.015)),
               borrow_fee=float(settings.get("frais d'emprunt de titres", 0.005)),
               cfd_overnight=float(settings.get("financement CFD", 0.025)),
               maintenance_margin=float(settings.get("marge de maintenance", 0.25 if leverage > 1 or short else 0.0)),
               min_trade=float(settings.get("ordre minimum", 0.0)))
    if isinstance(cash, str):
        cfg["cash_prices"] = cash_prices(cash)
    else:
        cfg["cash_rate"] = float(cash)
    cfg.update(management_fee=float(settings.get("frais de gestion", 0.0)),
               performance_fee=float(settings.get("commission de performance", 0.0)),
               triggers=tuple(ex.Trigger(d["type"], float(d["seuil"])) for d in settings.get("déclencheurs", [])))
    lines = settings.get("instruments") or atelier_instruments(universe)
    kinds = {r["ticker"]: ex.Instrument(r["ticker"], r["type"], ttf=bool(r.get("TTF")), stamp=bool(r.get("droit de timbre")),
                                        currency=str(r.get("devise") or "EUR"))
             for r in lines if r["ticker"] in panel.columns}
    advanced = ex.run_advanced(panel, rule, ex.AdvancedConfig(**cfg), instruments=kinds, label="Atelier")
    plain = ex.run_advanced(panel, rule, ex.AdvancedConfig(**{**cfg, "costs": ex.CostModel(0.001), "leverage": 1.0,
                                                              "max_gross": 1.0, "maintenance_margin": 0.0,
                                                              "management_fee": 0.0, "performance_fee": 0.0,
                                                              "triggers": ()}),
                            label="Laboratoire") if not short else None
    navs = {"Atelier (vos réglages)": advanced.nav}
    if plain is not None:
        navs["Même stratégie, frais du laboratoire, sans levier"] = plain.nav
    start = advanced.result.decisions["execution_date"].iloc[0] if not advanced.result.decisions.empty else panel.index[0]
    curves = pd.DataFrame(navs).loc[start:]
    curves = curves / curves.iloc[0] * 100
    rows = {}
    for label, nav in navs.items():
        s = summary(nav.loc[start:])
        rows[label] = {"Rendement annuel": s["cagr"], "Volatilité": s["volatility"], "Sharpe": s["sharpe"],
                       "Pire chute": s["max_drawdown"]}
    n = _atelier_log(ws, {"univers": universe, "profil": profile, "choix": choice, "réglages": settings,
                          "jeu": lab.dataset},
                     {"stratégie": name, "univers": universe, "profil": profile,
                      "réglages": {k: v for k, v in settings.items() if k != "instruments"},
                      "rendement annuel": rows["Atelier (vos réglages)"]["Rendement annuel"]})
    return {"metrics": pd.DataFrame(rows), "curves": curves, "costs": ex.cost_table(advanced),
            "events": advanced.events, "ruined": advanced.ruined, "n_configurations": n,
            "trades": advanced.result.trades,
            "exposure": advanced.result.weights.abs().sum(axis=1).loc[start:],
            "period": (str(start.date()), str(panel.index[-1].date()))}


# ------------------------------------------------------------------------- tracking
def tracker(ws: Workspace) -> Tracker:
    from . import code_rules
    return Tracker(ws.lab_dir / "suivi", rules=code_rules.load_all(ws.lab_dir))


def track_label(name: str, universe: str, profile: str | None) -> str:
    label = name if universe == "demo" else f"{name} · {universe}"
    return label + (f" · {profile}" if profile else "")


def start_tracking(ws: Workspace, name: str, universe: str, profile: str | None) -> dict:
    journal = ws.root / universes.get(universe)["journal"] / "registry.jsonl"
    return tracker(ws).start(frozen_from_journal(journal, name, profile=profile),
                             label=track_label(name, universe, profile),
                             source={"univers": universe, "profil": profile},
                             note="suivi démarré depuis l'interface")


def checkpoint(ws: Workspace) -> list[dict]:
    t = tracker(ws)
    out = []
    for name in t.tracked():
        universe = t.load(name).get("données", {}).get("univers", "demo")
        out.append(t.update(name, ws.prices(universe)))
    return out


# --------------------------------------------------------------------------- client
def frozen_records(ws: Workspace, universe: str, profile: str | None) -> dict[str, dict]:
    """Freeze records of a universe for a profile: the only strategies offered to a client."""
    j = ws.status()["journals"].get(universe)
    if not j:
        return {}
    return {f["name"]: f for f in j["freezes"] if (f.get("profile") or None) == (profile or None)}


def strategy_test(ws: Workspace, universe: str, profile: str | None, name: str) -> dict | None:
    """The test of a frozen strategy that counts (the first one; a REPEAT never counts), or None."""
    j = ws.status()["journals"].get(universe)
    if not j:
        return None
    tests = sorted((t for t in j["tests"] if t["name"] == name and (t["profile"] or None) == (profile or None)
                    and t["status"] != "REPEAT"), key=lambda t: t["tested_at"])
    if not tests:
        return None
    t = tests[0]
    return {"verdict": t["oos"].get("verdict"), "status": t["status"], "gap": t["oos"].get("ce3_gap_vs_1N"),
            "tested_at": t["tested_at"][:10]}


def diagnose_client(ws: Workspace, rows, cash: float, universe: str, profile: str | None,
                    strategy: dict | None, years: int | None):
    from .client import diagnose, read_lines
    lines, extra_cash = read_lines(rows)
    return diagnose(lines, cash=cash + extra_cash, profile=profile, universe=universe, strategy=strategy,
                    years=years, cache_dir=ws.cache_dir,
                    universe_loader=lambda name, cache_dir, verbose: ws.prices(name))


# ------------------------------------------------------------------- model portfolios
def model_targets(ws: Workspace, models: dict[str, dict]) -> tuple[dict[str, dict], dict[str, str]]:
    """Today's target weights of every model, and the reason a model has none."""
    from . import book
    targets, errors = {}, {}
    for name, m in models.items():
        try:
            u = m.get("univers") or "monde"
            if m["source"] == "manuel":
                targets[name] = book.model_weights(m)
            elif m["source"] == "référence":
                targets[name] = book.model_weights(m, universe_assets=universes.assets(u))
            else:
                record = frozen_records(ws, u, m["profil"]).get(m["stratégie"])
                targets[name] = book.model_weights(m, strategy_record=record,
                                                   uprices=ws.prices(u) if record else None)
        except Exception as exc:                    # one broken model must not stop the review
            errors[name] = str(exc)
    return targets, errors


def book_prices(ws: Workspace, currencies: dict[str, str], start: str = "2005-01-01"
                ) -> tuple[dict[str, float], pd.DataFrame]:
    """Last price in euros and history of each ticker, fetched one by one (a missing code
    does not hide the others). Returns ({ticker: last price}, history panel)."""
    from .data import DataError
    from .market_data import fetch_prices
    series = {}
    for t, cur in currencies.items():
        try:
            p = fetch_prices([t], start=start, cache_dir=ws.cache_dir, verbose=False,
                             currencies={t: cur}, base_currency="EUR")
            series[t] = p[t]
        except (DataError, KeyError, ValueError):
            continue
    history = pd.DataFrame(series).sort_index() if series else pd.DataFrame()
    last = {t: float(s.dropna().iloc[-1]) for t, s in series.items() if s.notna().any()}
    return last, history


def book_currencies(clients: dict[str, dict], models: dict[str, dict], targets: dict[str, dict]) -> dict[str, str]:
    """The currency of every code held by a client or targeted by a model."""
    from .client import guess_currency
    known = {}
    for name in universes.names():
        try:
            u = universes.get(name)
        except Exception:
            continue
        for a in u["actifs"] + u.get("recent", []):
            known.setdefault(a["ticker"], a.get("devise", u["devise"]))
    out = {}
    for c in clients.values():
        for line in (c.get("portefeuille") or {}).get("lignes", []):
            out[line["ticker"]] = line.get("devise") or known.get(line["ticker"]) or guess_currency(line["ticker"])
    for w in targets.values():
        for t in w:
            if t != "Cash":
                out.setdefault(t, known.get(t) or guess_currency(t))
    return out


# ------------------------------------------------------------------------ crisis tests
def client_crisis(ws: Workspace, d) -> pd.DataFrame:
    """Past crises replayed on a diagnosed portfolio (see stress.py)."""
    from . import stress
    weights = {t: float(w) for t, w in d.holdings["poids"].items()}
    currencies = {t: c for t, c in d.holdings["devise"].items() if t != "Cash"}
    for t in list(currencies):
        if t in stress.PROXIES:
            proxy, cur = stress.PROXIES[t]
            currencies.setdefault(proxy, cur)
    _, history = book_prices(ws, currencies, start="2005-01-01")
    return stress.crisis_tests(weights, history, d.total, d.profile.max_loss if d.profile else None)


# ------------------------------------------------------------------------ periodic report
def periodic_inputs(ws: Workspace, record: dict, models: dict[str, dict], start, end) -> dict:
    """The real track of a client's portfolio and of the allocation he was advised, same days."""
    from . import book, periodic
    from .client import guess_currency
    from .engine import BacktestConfig, run_backtest
    from .strategies import fixed_weights
    snaps = record.get("historique portefeuille") or ([record["portefeuille"]] if record.get("portefeuille") else [])
    currencies = {}
    for s in snaps:
        for line in s.get("lignes", []):
            if line["ticker"] != "CASH":
                currencies[line["ticker"]] = line.get("devise") or guess_currency(line["ticker"])
    _, history = book_prices(ws, currencies)
    if history.empty:
        raise ValueError("aucun prix pour les lignes du client")
    track = periodic.portfolio_track(snaps, history, start, end)
    model_name = (record.get("portefeuille") or {}).get("modèle")
    model = models.get(model_name) if model_name else None
    comparison, note = None, ""
    if model:
        u = model.get("univers") or "monde"
        try:
            if model["source"] == "stratégie":
                fr = frozen_records(ws, u, model["profil"]).get(model["stratégie"])
                nav = book.strategy_nav(fr, ws.prices(u)) if fr else None
                note = "la stratégie gelée suivie, avec ses vraies décisions jour après jour"
            else:
                targets, _ = model_targets(ws, {model_name: model})
                w = targets[model_name]
                _, h = book_prices(ws, book_currencies({}, {model_name: model}, {model_name: w}))
                h = h[[t for t in w]].dropna().loc[track["début"] - pd.Timedelta(days=10):]
                nav = run_backtest(h, fixed_weights(w), BacktestConfig(rebalance="monthly", lookback=0)).nav
                note = "les poids du modèle, rééquilibrés chaque mois, frais de 0,10 % inclus"
            if nav is not None:
                nav = nav.loc[track["début"]:track["fin"]]
                if len(nav) > 1:
                    comparison = nav / nav.iloc[0] * 100
        except Exception as exc:              # the report still goes out, without the comparison
            note = f"comparaison indisponible ({exc})"
    return {"track": track, "stats": periodic.stats(track["indice"]), "model": model, "model_name": model_name,
            "comparison": comparison, "comparison_stats": periodic.stats(comparison) if comparison is not None
            else None, "comparison_note": note}
