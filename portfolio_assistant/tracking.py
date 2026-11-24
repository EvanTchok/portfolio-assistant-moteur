"""Live tracking (paper trading) of a frozen strategy: the only test nobody can have seen.

A frozen strategy is registered on a given day. From the NEXT trading day on, every
update replays it on the prices that have appeared since, next to 1/N, and records the
result in an append-only journal. Nothing before the start date counts.

- the record keeps the frozen settings and the fingerprint of the code that runs them:
  if the code changes later, every update says so (status CODE_MODIFIÉ);
- each update appends a line to ``journal.jsonl``: what was reported on which day stays
  on file, even if the data provider later revises its prices;
- the portfolio on the start date is the one the strategy held at that date (a model
  portfolio carried forward), so no entry fee is charged on day one.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from .compare import compare
from .engine import run_backtest
from .research import StrategySpec, _source_hash
from .strategies import equal_weight

DEFAULT_FOLDER = "laboratoire/suivi"
MIN_MONTHS = 12          # below this, a gap with 1/N says almost nothing


def _slug(name: str) -> str:
    return re.sub(r"[^\w-]+", "_", name, flags=re.UNICODE).strip("_")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _spec_hash(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


def frozen_from_journal(journal: str | Path, name: str, profile: str | None = None) -> dict:
    """The freeze record of ``name`` (for ``profile``) in a lab journal (any data set)."""
    journal = Path(journal)
    if not journal.exists():
        raise FileNotFoundError(f"no lab journal at {journal}")
    freezes = [e for e in map(json.loads, journal.read_text(encoding="utf-8").splitlines())
               if e.get("event") == "freeze" and e.get("name") == name and e.get("profile") == profile]
    if not freezes:
        who = f" pour le profil {profile}" if profile else ""
        raise ValueError(f"« {name} » n'est pas gelée{who} dans {journal} : on ne suit qu'une stratégie gelée.")
    return freezes[-1]


class Tracker:
    """Registers frozen strategies and updates their live track record."""

    def __init__(self, folder: str | Path = DEFAULT_FOLDER, rules: dict | None = None):
        from .builder import composite
        from .strategies import RULES

        self.folder = Path(folder)
        package = Path(__file__).resolve().parent
        if package in self.folder.resolve().parents or self.folder.resolve() == package:
            raise ValueError("keep the tracking folder outside the package")
        self.folder.mkdir(parents=True, exist_ok=True)
        self.journal = self.folder / "journal.jsonl"
        self.rules = {**RULES, "composite": composite, **(rules or {})}

    # ------------------------------------------------------------------ registry
    def _path(self, name: str) -> Path:
        return self.folder / f"{_slug(name)}.json"

    def _append(self, event: dict) -> None:
        with self.journal.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"at": _now(), **event}, default=str, ensure_ascii=False) + "\n")

    def tracked(self) -> list[str]:
        return sorted(json.loads(p.read_text(encoding="utf-8"))["nom"]
                      for p in self.folder.glob("*.json"))

    def load(self, name: str) -> dict:
        path = self._path(name)
        if not path.exists():
            raise ValueError(f"« {name} » n'est pas suivie. Suivies : {', '.join(self.tracked()) or 'aucune'}")
        return json.loads(path.read_text(encoding="utf-8"))

    def start(self, frozen: dict, *, today: dt.date | None = None, note: str = "",
              label: str | None = None, source: dict | None = None) -> dict:
        """Starts tracking a frozen strategy (a freeze record from the lab journal).

        The first day that counts is the trading day AFTER ``today``: today's close may
        already be known when the tracking is registered. ``label`` names the track when
        the same strategy name is frozen in several universes; ``source`` records where
        its prices come from (e.g. {"univers": "pea"})."""
        name = label or frozen["name"]
        path = self._path(name)
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["empreinte stratégie"] != _spec_hash(frozen["spec"]):
                raise ValueError(f"« {name} » est déjà suivie avec une autre configuration.")
            return record
        today = today or dt.date.today()
        first_day = (pd.Timestamp(today) + pd.offsets.BDay(1)).date()
        record = {
            "nom": name,
            "spec": frozen["spec"],
            "empreinte stratégie": _spec_hash(frozen["spec"]),
            "empreinte du gel": frozen.get("fingerprint"),
            "code au gel": frozen.get("rule_code"),
            "enregistrée le": _now(),
            "premier jour compté": first_day.isoformat(),
            "note": note,
            "données": source or {},
        }
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        self._append({"event": "start", "nom": name, "premier jour compté": record["premier jour compté"],
                      "empreinte stratégie": record["empreinte stratégie"]})
        return record

    # ------------------------------------------------------------------ update
    def update(self, name: str, prices: pd.DataFrame, *, log: bool = True) -> dict:
        record = self.load(name)
        spec = StrategySpec.from_dict(record["spec"])
        factory = self.rules[spec.rule]
        rule = factory(**spec.params)
        code_now = _source_hash(factory)
        status = "INTACT" if record.get("code au gel") in (None, code_now) else "CODE_MODIFIÉ"

        first_day = pd.Timestamp(record["premier jour compté"])
        px = prices[list(spec.universe)] if spec.universe else prices
        new_days = px.index[px.index >= first_day]
        out = {"nom": name, "statut": status, "premier jour compté": first_day.date(),
               "derniers prix": px.index[-1].date(), "jours de bourse suivis": int(len(new_days))}
        if len(new_days) == 0:
            out["message"] = (f"le suivi compte à partir du {first_day.date()} et aucune séance n'a "
                              "encore eu lieu depuis : revenez après la prochaine clôture.")
            return out

        config = spec.config()
        strat = run_backtest(px, rule, config, label=name)
        profile_name = record.get("données", {}).get("profil")
        if profile_name:                      # judged for a client profile, as when frozen
            from . import profiles
            from .builder import _universe
            profile = profiles.get(profile_name)
            ref_label, gamma = f"référence {profile.label}", profile.gamma
            ref_rule = profiles.reference_rule(profile, _universe(spec.params["definition"]))()
        else:
            ref_label, gamma, ref_rule = "1/N", 3.0, equal_weight()
        base = run_backtest(px, ref_rule, config, label=ref_label)
        anchor = px.index[px.index < first_day][-1]           # close before the first day counted
        nav = pd.DataFrame({name: strat.nav, ref_label: base.nav}).loc[anchor:]
        growth = nav / nav.iloc[0] * 100
        cmp = compare(strat, base, start=anchor, gamma=gamma)
        months = (nav.index[-1] - anchor).days / 30.44
        perf = {"rendement depuis le début": growth.iloc[-1] / 100 - 1,
                "pire chute depuis le début": (growth / growth.cummax() - 1).min()}
        table = pd.DataFrame(perf).T
        table["écart"] = table[name] - table[ref_label]
        weights = strat.weights.iloc[-1]
        weights = weights[weights.abs() > 1e-9].sort_values(ascending=False)
        out.update({
            "mois suivis": months,
            "performance": table,
            "courbes (base 100)": growth,
            "poids actuels": weights,
            "cash actuel": float(1 - weights.sum()),
            "comparaison": cmp,
            "assez de recul": months >= MIN_MONTHS,
        })
        if log:
            self._append({"event": "update", "nom": name, "statut": status,
                          "derniers prix": out["derniers prix"],
                          "jours suivis": out["jours de bourse suivis"],
                          "rendement": float(table.loc["rendement depuis le début", name]),
                          "référence": ref_label,
                          "rendement 1/N": float(table.loc["rendement depuis le début", ref_label])})
        return out

    def history(self, name: str | None = None) -> pd.DataFrame:
        """Every update ever reported (append-only)."""
        if not self.journal.exists():
            return pd.DataFrame()
        rows = [e for e in map(json.loads, self.journal.read_text(encoding="utf-8").splitlines())
                if e["event"] == "update" and (name is None or e["nom"] == name)]
        return pd.DataFrame(rows)
