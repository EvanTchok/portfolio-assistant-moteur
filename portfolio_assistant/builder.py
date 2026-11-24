"""Build a strategy from blocks instead of writing code.

A strategy is a DEFINITION: plain data (a dict / JSON) that a form can fill in.

    {
      "nom": "Tendance + parité de risque",
      "univers": [{"ticker": "SPY", "classe": "actions"}, {"ticker": "AGG", "classe": "obligations"}, ...],
      "selection": [{"type": "trend", "days": 200}],
      "ponderation": {"type": "risk_parity", "days": 126},
      "contraintes": {"max_weight": 0.4, "max_per_class": {"actions": 0.6}, "min_cash": 0.0},
      "risque": {"target_vol": 0.10, "days": 63, "refuge": "AGG"},
      "rebalancement": "monthly",
      "frais": 0.001
    }

At every rebalancing date, using ONLY prices known up to that date:

    1. selection    each block keeps a subset of the eligible assets (applied in order)
    2. weighting    splits the money between the survivors
    3. constraints  max per line, max per asset class, minimum cash
    4. risk         volatility target (scales exposure down, never levers up),
                    refuge asset when nothing survives the selection

``catalogue()`` describes every block, its French label, explanation and parameter
ranges: it is what a user interface renders as a form. ``validate()`` returns the list
of problems in a definition, in French, before anything runs.
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .engine import DecisionContext

TRADING_DAYS = 252


# =============================================================================== catalogue
@dataclass(frozen=True)
class Param:
    name: str
    label: str
    kind: str                     # "int" | "float" | "percent" | "ticker" | "tickers" | "choice"
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    help: str = ""
    choices: tuple = ()

    def as_dict(self) -> dict:
        out = {"nom": self.name, "libellé": self.label, "type": self.kind, "défaut": self.default,
               "aide": self.help}
        if self.minimum is not None:
            out["min"] = self.minimum
        if self.maximum is not None:
            out["max"] = self.maximum
        if self.choices:
            out["choix"] = list(self.choices)
        return out


@dataclass(frozen=True)
class Block:
    kind: str                     # "selection" | "weighting"
    id: str
    label: str
    description: str
    params: tuple[Param, ...] = ()
    fn: Callable | None = field(default=None, compare=False, repr=False)

    def as_dict(self) -> dict:
        return {"id": self.id, "libellé": self.label, "description": self.description,
                "paramètres": [p.as_dict() for p in self.params]}


DAYS = lambda default, label="Période d'observation (jours de bourse)": Param(  # noqa: E731
    "days", label, "int", default, 20, 756, "252 jours de bourse ≈ 1 an, 126 ≈ 6 mois, 63 ≈ 3 mois")


# ------------------------------------------------------------------- estimation helpers
def _returns(prices: pd.DataFrame, days: int) -> pd.DataFrame:
    return prices.iloc[-(days + 1):].pct_change().iloc[1:]


def _covariance(prices: pd.DataFrame, days: int, shrink: bool = True) -> np.ndarray:
    """Daily covariance, shrunk towards a scaled identity (Ledoit & Wolf 2004).

    The sample covariance of 10 assets over 6 months is noisy, and optimisers amplify
    the noise. Shrinkage pulls it towards a stable target by an amount estimated from
    the data itself.
    """
    x = _returns(prices, days).to_numpy()
    x = x - x.mean(axis=0)
    t, n = x.shape
    sample = x.T @ x / t
    if not shrink or t < 2 or n < 2:
        return sample
    mu = np.trace(sample) / n
    target = mu * np.eye(n)
    d2 = np.sum((sample - target) ** 2)
    if d2 <= 0:
        return sample
    b2_bar = sum(np.sum((np.outer(row, row) - sample) ** 2) for row in x) / t**2
    delta = min(b2_bar, d2) / d2
    return delta * target + (1 - delta) * sample


# ------------------------------------------------------------------------ selection
def _sel_top_momentum(prices, eligible, days=252, k=3, skip=21):
    sub = prices[eligible]
    if len(sub) <= days:
        return eligible
    end = sub.iloc[-1 - skip] if skip else sub.iloc[-1]
    perf = (end / sub.iloc[-1 - days] - 1).sort_values(ascending=False)
    return list(perf.index[:int(k)])


def _sel_trend(prices, eligible, days=200):
    sub = prices[eligible]
    if len(sub) < days:
        return eligible
    above = sub.iloc[-1] > sub.iloc[-days:].mean()
    return [t for t in eligible if above[t]]


def _sel_absolute_momentum(prices, eligible, days=252, threshold=0.0):
    sub = prices[eligible]
    if len(sub) <= days:
        return eligible
    perf = sub.iloc[-1] / sub.iloc[-1 - days] - 1
    return [t for t in eligible if perf[t] > threshold]


def _sel_max_volatility(prices, eligible, days=63, max_vol=0.25):
    vol = _returns(prices[eligible], days).std() * math.sqrt(TRADING_DAYS)
    return [t for t in eligible if vol[t] <= max_vol]


def _sel_lowest_volatility(prices, eligible, days=63, k=3):
    vol = _returns(prices[eligible], days).std().sort_values()
    return list(vol.index[:int(k)])


def _sel_exclude(prices, eligible, tickers=()):
    banned = {str(t).upper() for t in tickers}
    return [t for t in eligible if t not in banned]


SELECTIONS: dict[str, Block] = {b.id: b for b in [
    Block("selection", "top_momentum", "Les meilleurs sur la période",
          "Garde les K actifs qui ont le plus monté sur la période (en sautant le dernier mois, "
          "qui a tendance à se retourner). C'est l'effet momentum.",
          (DAYS(252), Param("k", "Nombre d'actifs gardés", "int", 3, 1, 50),
           Param("skip", "Jours récents ignorés", "int", 21, 0, 63,
                 "Le dernier mois est souvent bruité ; 21 jours ≈ 1 mois")), _sel_top_momentum),
    Block("selection", "trend", "Au-dessus de sa tendance",
          "Garde un actif seulement si son prix est au-dessus de sa moyenne des N derniers jours. "
          "Classique : moyenne 200 jours. Sort des actifs en baisse durable.",
          (DAYS(200, "Longueur de la moyenne (jours)"),), _sel_trend),
    Block("selection", "absolute_momentum", "Rendement positif sur la période",
          "Garde un actif seulement si son rendement sur la période dépasse un seuil (0 % par défaut, "
          "ou le taux du cash). Combiné au précédent, cela donne le « momentum dual ».",
          (DAYS(252), Param("threshold", "Seuil de rendement", "percent", 0.0, -0.5, 0.5)),
          _sel_absolute_momentum),
    Block("selection", "max_volatility", "Volatilité plafonnée",
          "Écarte les actifs dont la volatilité récente (annualisée) dépasse un plafond.",
          (DAYS(63), Param("max_vol", "Volatilité maximale", "percent", 0.25, 0.01, 1.0)),
          _sel_max_volatility),
    Block("selection", "lowest_volatility", "Les plus calmes",
          "Garde les K actifs les moins volatils sur la période.",
          (DAYS(63), Param("k", "Nombre d'actifs gardés", "int", 3, 1, 50)), _sel_lowest_volatility),
    Block("selection", "exclude", "Exclure des actifs",
          "Retire des actifs de la sélection (choix du gérant ou du client).",
          (Param("tickers", "Actifs exclus", "tickers", []),), _sel_exclude),
]}


# ------------------------------------------------------------------------ weighting
@dataclass
class _Frame:
    """What an optimiser must respect: bounds per asset, caps per class, total invested."""
    tickers: list
    lower: np.ndarray
    upper: np.ndarray
    class_caps: list            # [(mask array, cap)]
    total: float

    def scipy_constraints(self):
        cons = [{"type": "eq", "fun": lambda w, s=self.total: np.sum(w) - s}]
        for mask, cap in self.class_caps:
            cons.append({"type": "ineq", "fun": lambda w, m=mask, c=cap: c - np.sum(w[m])})
        return cons

    def feasible_total(self) -> float:
        """The most that can be invested under the caps; the rest has to stay in cash."""
        cap_lines = float(np.sum(self.upper))
        if self.class_caps:
            unclassed = np.ones(len(self.tickers), dtype=bool)
            by_class = 0.0
            for mask, cap in self.class_caps:
                by_class += min(cap, float(np.sum(self.upper[mask])))
                unclassed &= ~mask
            cap_lines = min(cap_lines, by_class + float(np.sum(self.upper[unclassed])))
        return min(self.total, cap_lines)


def _solve(objective, start: np.ndarray, frame: _Frame) -> np.ndarray:
    total = frame.feasible_total()
    frame = _Frame(frame.tickers, frame.lower, frame.upper, frame.class_caps, total)
    x0 = np.clip(start, frame.lower, frame.upper)
    if x0.sum() > 0:
        x0 = x0 * total / x0.sum()
    res = minimize(objective, x0, method="SLSQP", bounds=list(zip(frame.lower, frame.upper)),
                   constraints=frame.scipy_constraints(), options={"maxiter": 500, "ftol": 1e-12})
    if res.success or _acceptable(res.x, x0, objective, frame):
        return np.clip(res.x, frame.lower, frame.upper)
    return np.clip(x0, frame.lower, frame.upper)


def _acceptable(x: np.ndarray, x0: np.ndarray, objective, frame: _Frame, tol: float = 1e-7) -> bool:
    """SLSQP sometimes stops on a numerical stall ("positive directional derivative") right
    at the optimum. Its point is then kept if it respects every constraint and does at
    least as well as the start; otherwise falling back to the start would silently swap
    the strategy for 1/N that month."""
    if not np.all(np.isfinite(x)):
        return False
    feasible = (np.all(x >= frame.lower - tol) and np.all(x <= frame.upper + tol)
                and abs(x.sum() - frame.total) < tol
                and all(np.sum(x[m]) <= c + tol for m, c in frame.class_caps))
    return bool(feasible and objective(x) <= objective(x0) + 1e-12)


def _project(target: np.ndarray, frame: _Frame) -> np.ndarray:
    """Closest weights to ``target`` that respect every constraint (least squares)."""
    if not len(target):
        return target
    already_ok = (np.all(target <= frame.upper + 1e-12) and np.all(target >= frame.lower - 1e-12)
                  and all(np.sum(target[m]) <= c + 1e-12 for m, c in frame.class_caps)
                  and abs(target.sum() - frame.total) < 1e-12)
    if already_ok:
        return target
    return _solve(lambda w: np.sum((w - target) ** 2), target, frame)


def _w_equal(prices, frame, **_):
    n = len(frame.tickers)
    return _project(np.full(n, frame.total / n), frame)


def _w_inverse_vol(prices, frame, days=63, **_):
    vol = _returns(prices[frame.tickers], days).std().to_numpy()
    inv = np.where(vol > 0, 1 / np.where(vol > 0, vol, 1), 0)
    if inv.sum() <= 0:
        return _w_equal(prices, frame)
    return _project(inv / inv.sum() * frame.total, frame)


def _w_min_variance(prices, frame, days=126, **_):
    cov = _covariance(prices[frame.tickers], days) * TRADING_DAYS
    n = len(frame.tickers)
    return _solve(lambda w: w @ cov @ w * 1e4, np.full(n, frame.total / n), frame)


def _w_risk_parity(prices, frame, days=126, **_):
    """Each asset contributes the same share of portfolio risk (Spinu's convex form)."""
    cov = _covariance(prices[frame.tickers], days) * TRADING_DAYS
    n = len(frame.tickers)
    budget = np.full(n, 1 / n)
    res = minimize(lambda y: 0.5 * y @ cov @ y - budget @ np.log(y), np.full(n, 1 / n),
                   jac=lambda y: cov @ y - budget / y, method="L-BFGS-B",
                   bounds=[(1e-10, None)] * n)
    y = res.x if res.success else 1 / np.sqrt(np.diag(cov))
    return _project(y / y.sum() * frame.total, frame)


def _w_max_sharpe(prices, frame, days=252, risk_free=0.0, **_):
    """Markowitz tangency portfolio, from historical means (noisy: use with care)."""
    sub = prices[frame.tickers]
    cov = _covariance(sub, days) * TRADING_DAYS
    mu = _returns(sub, days).mean().to_numpy() * TRADING_DAYS - risk_free
    n = len(frame.tickers)
    if np.all(mu <= 0):  # no asset is expected to beat cash: least risky portfolio instead
        return _w_min_variance(prices, frame, days=days)

    def neg_sharpe(w):
        vol = math.sqrt(max(w @ cov @ w, 1e-16))
        return -(w @ mu) / vol

    return _solve(neg_sharpe, np.full(n, frame.total / n), frame)


def _w_fixed(prices, frame, weights=None, **_):
    """The manager's weights. A line removed by the selection keeps its share in CASH:
    the others are not inflated to fill the gap (that is what a trend filter means)."""
    weights = {str(k).upper(): float(v) for k, v in (weights or {}).items()}
    configured = sum(weights.values())
    scale = frame.total / configured if configured > 1 else frame.total
    raw = np.array([weights.get(t, 0.0) for t in frame.tickers]) * scale
    if raw.sum() <= 0:
        return np.zeros(len(frame.tickers))
    kept = _Frame(frame.tickers, np.minimum(frame.lower, raw), frame.upper, frame.class_caps,
                  float(raw.sum()))
    return _project(raw, kept)


def _w_momentum_proportional(prices, frame, days=252, **_):
    sub = prices[frame.tickers]
    if len(sub) <= days:
        return _w_equal(prices, frame)
    perf = (sub.iloc[-1] / sub.iloc[-1 - days] - 1).clip(lower=0).to_numpy()
    if perf.sum() <= 0:
        return _w_equal(prices, frame)
    return _project(perf / perf.sum() * frame.total, frame)


WEIGHTINGS: dict[str, Block] = {b.id: b for b in [
    Block("weighting", "equal", "Poids égaux (1/N)",
          "Chaque actif retenu reçoit la même part. Simple et souvent difficile à battre.", (), _w_equal),
    Block("weighting", "inverse_vol", "Inverse de la volatilité",
          "Plus de poids aux actifs calmes, moins aux actifs agités.", (DAYS(63),), _w_inverse_vol),
    Block("weighting", "min_variance", "Variance minimale",
          "Le portefeuille le moins volatil possible, en tenant compte des corrélations entre actifs.",
          (DAYS(126),), _w_min_variance),
    Block("weighting", "risk_parity", "Parité de risque",
          "Chaque actif apporte la même part du risque total du portefeuille (approche « All Weather »).",
          (DAYS(126),), _w_risk_parity),
    Block("weighting", "max_sharpe", "Markowitz (Sharpe maximal)",
          "Le meilleur rendement par unité de risque, estimé sur le passé. Puissant mais sensible aux "
          "erreurs d'estimation : toujours le comparer aux méthodes simples.",
          (DAYS(252), Param("risk_free", "Taux sans risque annuel", "percent", 0.0, 0.0, 0.2)),
          _w_max_sharpe),
    Block("weighting", "momentum_proportional", "Proportionnel au momentum",
          "Poids proportionnel à la hausse de chaque actif sur la période (les actifs en baisse reçoivent 0).",
          (DAYS(252),), _w_momentum_proportional),
    Block("weighting", "fixed", "Allocation fixe",
          "Les poids choisis par le gérant (par exemple 60 % actions / 40 % obligations), "
          "rétablis à chaque rebalancement.",
          (Param("weights", "Poids par actif", "weights", {}),), _w_fixed),
]}

CONSTRAINT_PARAMS = (
    Param("max_weight", "Poids maximal par ligne", "percent", 1.0, 0.01, 1.0),
    Param("min_weight", "Poids minimal par ligne retenue", "percent", 0.0, 0.0, 0.5),
    Param("max_per_class", "Poids maximal par classe d'actifs", "class_caps", {}),
    Param("min_cash", "Cash minimal", "percent", 0.0, 0.0, 0.9),
)
RISK_PARAMS = (
    Param("target_vol", "Cible de volatilité (vide = aucune)", "percent", None, 0.01, 0.5,
          "Réduit l'exposition quand le marché s'agite ; jamais de levier."),
    Param("days", "Période d'estimation de la volatilité (jours)", "int", 63, 20, 252),
    Param("refuge", "Actif refuge si rien n'est sélectionné (vide = cash)", "ticker", None),
)
REBALANCE_CHOICES = ("weekly", "monthly", "quarterly", "yearly")


def catalogue() -> dict:
    """Everything a user interface needs to build the strategy form."""
    return {
        "sélection": [b.as_dict() for b in SELECTIONS.values()],
        "pondération": [b.as_dict() for b in WEIGHTINGS.values()],
        "contraintes": [p.as_dict() for p in CONSTRAINT_PARAMS],
        "risque": [p.as_dict() for p in RISK_PARAMS],
        "rebalancement": {"choix": list(REBALANCE_CHOICES), "défaut": "monthly",
                          "libellés": {"weekly": "Chaque semaine", "monthly": "Chaque mois",
                                       "quarterly": "Chaque trimestre", "yearly": "Chaque année"}},
        "frais": Param("frais", "Frais par transaction", "percent", 0.001, 0.0, 0.02).as_dict(),
    }


# =============================================================================== validation
def _universe(defn: dict) -> list[dict]:
    out = []
    for item in defn.get("univers", []):
        if isinstance(item, str):
            item = {"ticker": item}
        out.append({"ticker": str(item["ticker"]).strip().upper(),
                    "classe": str(item.get("classe") or "autre").strip().lower()})
    return out


def _check_params(block: Block, given: dict, where: str, errors: list) -> None:
    known = {p.name: p for p in block.params}
    for name in given:
        if name not in known and name != "type":
            errors.append(f"{where} « {block.label} » : réglage inconnu « {name} »")
    for p in block.params:
        value = given.get(p.name, p.default)
        if p.kind in ("int", "float", "percent") and value is not None:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                errors.append(f"{where} « {block.label} » : « {p.label} » doit être un nombre")
                continue
            if p.kind == "int" and int(value) != value:
                errors.append(f"{where} « {block.label} » : « {p.label} » doit être un entier")
            if p.minimum is not None and value < p.minimum:
                errors.append(f"{where} « {block.label} » : « {p.label} » doit être ≥ {p.minimum}")
            if p.maximum is not None and value > p.maximum:
                errors.append(f"{where} « {block.label} » : « {p.label} » doit être ≤ {p.maximum}")


def validate(defn: dict, available: list[str] | None = None) -> list[str]:
    """Every problem in a definition, in French. An empty list means it can run."""
    errors: list[str] = []
    if not str(defn.get("nom", "")).strip():
        errors.append("la stratégie doit avoir un nom")
    universe = _universe(defn)
    tickers = [u["ticker"] for u in universe]
    if len(tickers) < 1:
        errors.append("l'univers doit contenir au moins un actif")
    if len(set(tickers)) != len(tickers):
        errors.append("un actif apparaît deux fois dans l'univers")
    if available is not None:
        missing = sorted(set(tickers) - {a.upper() for a in available})
        if missing:
            errors.append(f"actifs sans données de prix : {', '.join(missing)}")
    for i, sel in enumerate(defn.get("selection", []), start=1):
        block = SELECTIONS.get(sel.get("type"))
        if block is None:
            errors.append(f"sélection n°{i} : type inconnu « {sel.get('type')} » "
                          f"(possibles : {', '.join(SELECTIONS)})")
            continue
        _check_params(block, sel, f"sélection n°{i}", errors)
    weighting = defn.get("ponderation", {"type": "equal"})
    block = WEIGHTINGS.get(weighting.get("type"))
    if block is None:
        errors.append(f"pondération : type inconnu « {weighting.get('type')} » "
                      f"(possibles : {', '.join(WEIGHTINGS)})")
    else:
        _check_params(block, weighting, "pondération", errors)
        if block.id == "fixed":
            unknown = sorted({str(k).upper() for k in weighting.get("weights", {})} - set(tickers))
            if unknown:
                errors.append(f"allocation fixe : actifs hors univers {', '.join(unknown)}")
    cons = defn.get("contraintes", {})
    for p in CONSTRAINT_PARAMS:
        v = cons.get(p.name)
        if p.kind == "percent" and v is not None and not (p.minimum <= v <= p.maximum):
            errors.append(f"contraintes : « {p.label} » doit être entre {p.minimum:.0%} et {p.maximum:.0%}")
    if cons.get("min_weight", 0) > cons.get("max_weight", 1):
        errors.append("contraintes : le poids minimal dépasse le poids maximal")
    classes = {u["classe"] for u in universe}
    for name, cap in cons.get("max_per_class", {}).items():
        if str(name).lower() not in classes:
            errors.append(f"contraintes : la classe « {name} » n'existe pas dans l'univers "
                          f"(classes : {', '.join(sorted(classes))})")
        if not 0 <= cap <= 1:
            errors.append(f"contraintes : le plafond de la classe « {name} » doit être entre 0 % et 100 %")
    risk = defn.get("risque", {})
    if risk.get("target_vol") is not None and not 0.01 <= risk["target_vol"] <= 0.5:
        errors.append("risque : la cible de volatilité doit être entre 1 % et 50 %")
    refuge = risk.get("refuge")
    if refuge and str(refuge).upper() not in tickers:
        errors.append(f"risque : l'actif refuge {refuge} doit faire partie de l'univers")
    if defn.get("rebalancement", "monthly") not in REBALANCE_CHOICES:
        errors.append(f"rebalancement : choisir parmi {', '.join(REBALANCE_CHOICES)}")
    fee = defn.get("frais", 0.001)
    if not 0 <= fee <= 0.02:
        errors.append("frais : entre 0 % et 2 % par transaction")
    return errors


def required_history(defn: dict) -> int:
    """Bars of history needed before the first decision: the longest look-back used."""
    days = [int(s.get("days", next((p.default for p in SELECTIONS[s["type"]].params if p.name == "days"), 0)))
            for s in defn.get("selection", []) if s.get("type") in SELECTIONS]
    w = defn.get("ponderation", {"type": "equal"})
    if w.get("type") in WEIGHTINGS:
        default = next((p.default for p in WEIGHTINGS[w["type"]].params if p.name == "days"), 0)
        days.append(int(w.get("days", default)))
    if defn.get("risque", {}).get("target_vol") is not None:
        days.append(int(defn["risque"].get("days", 63)))
    skip = max([int(s.get("skip", 21)) for s in defn.get("selection", []) if s.get("type") == "top_momentum"] or [0])
    return max(days + [0]) + skip + 1


# ============================================================================ the strategy
def _block_kwargs(block: Block, given: dict) -> dict:
    out = {}
    for p in block.params:
        value = given.get(p.name, p.default)
        if p.kind == "int":
            value = int(value)
        out[p.name] = value
    return out


def composite(definition: dict):
    """The weighting rule built from a definition (the engine calls it at each decision)."""
    errors = validate(definition)
    if errors:
        raise ValueError("stratégie invalide :\n  - " + "\n  - ".join(errors))
    defn = copy.deepcopy(definition)
    universe = _universe(defn)
    tickers = [u["ticker"] for u in universe]
    classes = {u["ticker"]: u["classe"] for u in universe}
    selections = [(SELECTIONS[s["type"]], s) for s in defn.get("selection", [])]
    weighting = defn.get("ponderation", {"type": "equal"})
    w_block = WEIGHTINGS[weighting["type"]]
    cons = defn.get("contraintes", {})
    risk = defn.get("risque", {})
    refuge = str(risk["refuge"]).upper() if risk.get("refuge") else None
    caps = {str(k).lower(): float(v) for k, v in cons.get("max_per_class", {}).items()}

    def fn(ctx: DecisionContext):
        prices = ctx.history[tickers]
        eligible = list(tickers)
        for block, given in selections:
            eligible = block.fn(prices, eligible, **_block_kwargs(block, given))
            if not eligible:
                break
        total = 1.0 - float(cons.get("min_cash", 0.0))
        if not eligible:
            return {refuge: total} if refuge else {}
        frame = _Frame(
            tickers=eligible,
            lower=np.full(len(eligible), float(cons.get("min_weight", 0.0))),
            upper=np.full(len(eligible), float(cons.get("max_weight", 1.0))),
            class_caps=[(np.array([classes[t] == c for t in eligible]), cap) for c, cap in caps.items()
                        if any(classes[t] == c for t in eligible)],
            total=total,
        )
        if frame.lower.sum() > frame.total:          # minimum weights cannot all be honoured
            frame.lower = np.zeros(len(eligible))
        weights = w_block.fn(prices, frame, **_block_kwargs(w_block, weighting))
        weights = np.clip(np.nan_to_num(weights), 0.0, None)
        if risk.get("target_vol") and weights.sum() > 0:
            cov = _covariance(prices[eligible], int(risk.get("days", 63))) * TRADING_DAYS
            vol = math.sqrt(max(weights @ cov @ weights, 0.0))
            if vol > risk["target_vol"]:
                weights = weights * risk["target_vol"] / vol
        if weights.sum() > total + 1e-9:
            weights = weights * total / weights.sum()
        return {t: float(w) for t, w in zip(eligible, weights) if w > 1e-6}

    return fn


# The lab fingerprints a rule by its source code. A composite strategy depends on every
# block above, so its fingerprint is the whole module: editing any block is detected.
composite.__code_fingerprint__ = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def to_spec(definition: dict):
    """A lab StrategySpec carrying the definition, with the history it needs."""
    from .research import StrategySpec

    errors = validate(definition)
    if errors:
        raise ValueError("stratégie invalide :\n  - " + "\n  - ".join(errors))
    return StrategySpec(
        name=str(definition["nom"]).strip(), rule="composite", params={"definition": definition},
        universe=tuple(u["ticker"] for u in _universe(definition)),
        rebalance=definition.get("rebalancement", "monthly"),
        lookback=required_history(definition), fee_rate=float(definition.get("frais", 0.001)),
        description=str(definition.get("description", "")),
    )


def _get_path(defn: dict, path: str):
    node = defn
    for part in path.split("."):
        node = node[int(part)] if isinstance(node, list) else node.get(part)
    return node


def set_path(definition: dict, path: str, value) -> dict:
    """A copy of the definition with one setting changed, e.g. ``"selection.0.k"`` or
    ``"ponderation.days"``. This is how a parameter grid varies a block strategy."""
    out = copy.deepcopy(definition)
    parts = path.split(".")
    node = out
    for part in parts[:-1]:
        node = node[int(part)] if isinstance(node, list) else node.setdefault(part, {})
    last = parts[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value
    return out


def apply_settings(params: dict, settings: dict) -> dict:
    """Parameters with grid settings applied: nested paths for a block strategy,
    plain keys for a coded rule."""
    if "definition" in params:
        definition = params["definition"]
        for path, value in settings.items():
            definition = set_path(definition, path, value)
        return {**params, "definition": definition}
    return {**params, **settings}


def current_settings(params: dict, names) -> dict:
    if "definition" in params:
        out = {}
        for name in names:
            try:
                out[name] = _get_path(params["definition"], name)
            except (KeyError, IndexError, AttributeError, TypeError):
                out[name] = None
        return out
    return {n: params.get(n) for n in names}


def describe(definition: dict) -> str:
    """The strategy in plain French, block by block (for a report or a client)."""
    lines = [f"Stratégie « {definition.get('nom', '?')} »"]
    universe = _universe(definition)
    by_class: dict[str, list] = {}
    for u in universe:
        by_class.setdefault(u["classe"], []).append(u["ticker"])
    lines.append("Univers : " + " ; ".join(f"{c} : {', '.join(t)}" for c, t in by_class.items()))
    for i, sel in enumerate(definition.get("selection", []), start=1):
        block = SELECTIONS[sel["type"]]
        params = ", ".join(f"{p.label.lower()} = {sel.get(p.name, p.default)}" for p in block.params)
        lines.append(f"Sélection {i} : {block.label} ({params})")
    w = definition.get("ponderation", {"type": "equal"})
    block = WEIGHTINGS[w["type"]]
    params = ", ".join(f"{p.label.lower()} = {w.get(p.name, p.default)}" for p in block.params)
    lines.append(f"Pondération : {block.label}" + (f" ({params})" if params else ""))
    cons = definition.get("contraintes", {})
    if cons:
        parts = []
        if "max_weight" in cons:
            parts.append(f"{cons['max_weight']:.0%} maximum par ligne")
        if "min_weight" in cons:
            parts.append(f"{cons['min_weight']:.0%} minimum par ligne retenue")
        for c, cap in cons.get("max_per_class", {}).items():
            parts.append(f"{cap:.0%} maximum en {c}")
        if cons.get("min_cash"):
            parts.append(f"{cons['min_cash']:.0%} de cash minimum")
        lines.append("Contraintes : " + ", ".join(parts))
    risk = definition.get("risque", {})
    if risk.get("target_vol"):
        lines.append(f"Risque : volatilité ciblée à {risk['target_vol']:.0%} (sans levier)")
    if risk.get("refuge"):
        lines.append(f"Refuge si rien n'est sélectionné : {risk['refuge']}")
    lines.append(f"Rebalancement : {definition.get('rebalancement', 'monthly')}, "
                 f"frais {definition.get('frais', 0.001):.2%} par transaction")
    return "\n".join(lines)


# ================================================================================ storage
def save_definition(definition: dict, folder: str | Path = "strategies") -> Path:
    """Saves a strategy as a readable JSON file (one file per strategy)."""
    import json
    errors = validate(definition)
    if errors:
        raise ValueError("stratégie invalide :\n  - " + "\n  - ".join(errors))
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(definition["nom"]).strip())
    path = folder / f"{safe}.json"
    path.write_text(json.dumps(definition, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_definition(path: str | Path) -> dict:
    import json
    definition = json.loads(Path(path).read_text(encoding="utf-8"))
    errors = validate(definition)
    if errors:
        raise ValueError(f"{path} : stratégie invalide :\n  - " + "\n  - ".join(errors))
    return definition


# ================================================================================ presets
def presets(universe: list) -> dict[str, dict]:
    """The classic allocation strategies, ready to use and to modify, on a given universe.

    ``universe`` is a list of {"ticker", "classe"}; the first asset of class
    "obligations" (if any) is used as the refuge.
    """
    uni = _universe({"univers": universe})
    tickers = [u["ticker"] for u in uni]
    bonds = [u["ticker"] for u in uni if u["classe"] == "obligations"]
    refuge = bonds[0] if bonds else None
    n = len(tickers)
    base = {"univers": uni, "rebalancement": "monthly", "frais": 0.001}

    def make(name, description, **parts):
        return {"nom": name, "description": description, **copy.deepcopy(base), **parts}

    out = [
        make("Poids égaux", "La référence : 1/N sur tout l'univers, rebalancé chaque mois.",
             ponderation={"type": "equal"}),
        make("Variance minimale", "Le portefeuille le moins volatil, 50 % maximum par ligne.",
             ponderation={"type": "min_variance", "days": 126}, contraintes={"max_weight": 0.5}),
        make("Parité de risque", "Chaque actif apporte la même part du risque total.",
             ponderation={"type": "risk_parity", "days": 126}),
        make("Markowitz", "Sharpe maximal estimé sur 1 an, 40 % maximum par ligne pour limiter "
             "la concentration.", ponderation={"type": "max_sharpe", "days": 252},
             contraintes={"max_weight": 0.4}),
        make("Suivi de tendance", "Chaque actif a une part de 1/N ; il ne la garde que s'il est "
             "au-dessus de sa moyenne 200 jours, sinon cette part reste en cash (approche Faber).",
             selection=[{"type": "trend", "days": 200}],
             ponderation={"type": "fixed", "weights": {t: 1 / n for t in tickers}}),
        make("Momentum dual", "Le meilleur actif sur 12 mois, seulement si sa hausse est positive ; "
             "sinon tout va dans le refuge obligataire.",
             selection=[{"type": "top_momentum", "days": 252, "k": 1},
                        {"type": "absolute_momentum", "days": 252, "threshold": 0.0}],
             ponderation={"type": "equal"}, risque={"refuge": refuge} if refuge else {}),
        make("Cible de volatilité 10 %", "Poids égaux, exposition réduite quand la volatilité "
             "dépasse 10 % (le reste en cash).", ponderation={"type": "equal"},
             risque={"target_vol": 0.10, "days": 63}),
        make("Tendance + parité de risque", "Actifs au-dessus de leur moyenne 200 jours, répartis "
             "en parité de risque, 40 % maximum par ligne.",
             selection=[{"type": "trend", "days": 200}],
             ponderation={"type": "risk_parity", "days": 126}, contraintes={"max_weight": 0.4},
             risque={"refuge": refuge} if refuge else {}),
    ]
    return {d["nom"]: d for d in out}
