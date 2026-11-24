"""The twelve validation questions asked of a strategy before anyone relies on it.

    1  Référence            does it beat the reference of its profile on the same calendar?
    2  Covariance           does the risk it estimates match the risk that follows?
    3  Taille neutre        do its weights add anything over equal weights on the same picks?
    4  Levier               does it beat simply holding more or less of the reference?
    5  Concentration        is the gain spread over time, or one lucky episode?
    6  Monte-Carlo          does the gain survive reshuffled market paths?
    7  Pouvoir d'achat      what is left after inflation, and in inflationary months?
    8  Stress inversé       at what fall of the equity markets does it break its profile?
    9  Triage               is the sample long enough to conclude anything?
   10  Sharpe dégonflé      is the Sharpe real once every trial is counted?
   11  Hors échantillon     how much of the edge survived the sealed test (if it was opened)?
   12  Multi-actifs         does the edge hold on baskets of assets never used to build it?

Everything runs on the CONSTRUCTION period only, except test 11, which only READS a test
already recorded in the registry: this battery never opens the sealed period, and its
results are diagnostics, never counted as trials.

Each test returns a ``Verdict`` with a plain sentence and the figures behind it. The
thresholds are fixed here, before any strategy is examined.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .engine import run_backtest
from .metrics import certainty_equivalent, max_drawdown, periods_per_year

PASS, BORDER, FAIL, NA = "passe", "limite", "échoue", "non applicable"
EQUALITY = 0.0025                 # same rule as the lab: |gap| < 0.25 %/yr is no difference
INFLATION_FILE = Path(__file__).with_name("donnees") / "inflation_france.csv"
FINANCING_SPREAD = 0.015          # borrowing cost above cash to lever the reference (test 4)

# Test 12 draws its baskets from here: funds that no universe of the lab uses, kept for
# validation only, so that no strategy can have been tuned on them. US listings, for their
# long histories (most start before 2008); prices are converted to euros day by day.
RESERVOIR = {
    "actions": ["IWM", "QQQ", "DIA", "MDY", "EFA", "VGK", "EWG", "EWU", "EWQ", "EWL", "EWC", "EWA", "EWZ",
                "FXI", "EWT", "EWY", "XLK", "XLF", "XLE", "XLV", "XLP", "XLU", "XLI"],
    "obligations": ["TLT", "IEF", "SHY", "LQD", "TIP", "AGG", "BND", "HYG", "EMB", "BWX"],
    "or": ["IAU"],
    "immobilier": ["IYR", "RWR", "ICF"],
    "matières premières": ["GSG", "DBA", "USO", "DBB", "DBE"],
}


def reservoir_prices(cache_dir, start: str = "2005-01-01", base_currency: str = "EUR") -> tuple[pd.DataFrame, dict]:
    """The reservoir's prices in euros (each fund on its own dates; missing ones are skipped)."""
    from .data import DataError
    from .market_data import fetch_prices
    series, classes = {}, {}
    for cls, tickers in RESERVOIR.items():
        for t in tickers:
            try:
                p = fetch_prices([t], start=start, cache_dir=cache_dir, verbose=False,
                                 currencies={t: "USD"}, base_currency=base_currency)
                series[t], classes[t] = p[t], cls
            except (DataError, KeyError, ValueError):
                continue
    return (pd.DataFrame(series).sort_index() if series else pd.DataFrame()), classes


QUESTIONS = [
    (1, "Référence", "Fait-elle mieux que la référence de son profil, sur le même calendrier ?"),
    (2, "Prévision du risque", "Le risque estimé au moment de décider annonce-t-il le risque qui suit ?"),
    (3, "Taille neutre", "Ses pondérations apportent-elles quelque chose face à des parts égales sur les mêmes choix ?"),
    (4, "Levier", "Fait-elle mieux que la référence simplement allégée ou renforcée au même risque ?"),
    (5, "Concentration du gain", "Le gain est-il réparti dans le temps, ou dû à quelques mois chanceux ?"),
    (6, "Monte-Carlo", "Le gain résiste-t-il quand l'histoire est rebattue par blocs ?"),
    (7, "Pouvoir d'achat", "Que reste-t-il après l'inflation, y compris quand elle est forte ?"),
    (8, "Stress inversé", "À partir de quelle baisse des actions sort-elle des limites de son profil ?"),
    (9, "Triage", "L'échantillon est-il assez long pour conclure ?"),
    (10, "Sharpe dégonflé", "Son Sharpe reste-t-il significatif une fois tous les essais comptés ?"),
    (11, "Hors échantillon", "Quelle part de l'avantage a survécu au test scellé ?"),
    (12, "Multi-actifs", "L'avantage tient-il sur des paniers d'actifs jamais utilisés pour la construire ?"),
]


@dataclass
class Verdict:
    number: int
    title: str
    question: str
    status: str
    sentence: str
    figures: dict = field(default_factory=dict)
    table: pd.DataFrame | None = None

    def to_dict(self) -> dict:
        d = {"n": self.number, "test": self.title, "question": self.question, "verdict": self.status,
             "phrase": self.sentence, "chiffres": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                                   for k, v in self.figures.items()}}
        if self.table is not None:
            d["tableau"] = self.table.to_dict(orient="records")
        return d


def _pct(v: float, digits: int = 1, signed: bool = False) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "–"
    text = f"{v * 100:+.{digits}f}" if signed else f"{v * 100:.{digits}f}"
    return text.replace("-", "−").replace(".", ",") + " %"


def _n(v: float, digits: int = 2) -> str:
    return f"{v:.{digits}f}".replace("-", "−").replace(".", ",")


def _p0(v: float) -> str:
    return f"{v * 100:.0f}".replace("-", "−") + " %"


def _years(nav: pd.Series) -> float:
    return (nav.index[-1] - nav.index[0]).days / 365.25


def _cagr(nav: pd.Series) -> float:
    return float((nav.iloc[-1] / nav.iloc[0]) ** (1 / _years(nav)) - 1)


def _ce(nav: pd.Series, gamma: float) -> float:
    r = nav.pct_change().dropna()
    return certainty_equivalent(r, gamma, periods_per_year(nav.index))


def _first_exec(result) -> pd.Timestamp:
    return result.decisions["execution_date"].iloc[0] if not result.decisions.empty else result.nav.index[0]


def _gap_status(gap: float) -> str:
    return PASS if gap >= EQUALITY else (BORDER if gap > -EQUALITY else FAIL)


# =============================================================================== context
@dataclass
class Context:
    """What every test needs, computed once: the construction-period runs."""
    lab: object
    spec: object
    panel: pd.DataFrame
    config: object
    factory: Callable
    ref: object
    strategy: object          # BacktestResult
    baseline: object
    nav: pd.Series            # from the first execution of both
    base: pd.Series
    gamma: float


def context(lab, spec) -> Context:
    panel = lab._panel(spec, lab.is_prices)
    config = spec.config()
    factory = lab.rules[spec.rule]
    ref = lab.reference(spec)
    strategy = run_backtest(panel, factory(**spec.params), config, label=spec.name)
    baseline = run_backtest(panel, ref.rule(), config, label=ref.label)
    start = max(_first_exec(strategy), _first_exec(baseline))
    return Context(lab, spec, panel, config, factory, ref, strategy, baseline,
                   strategy.nav.loc[start:], baseline.nav.loc[start:], lab.gamma)


# =============================================================================== the tests
def t1_reference(c: Context) -> Verdict:
    gap = _ce(c.nav, c.gamma) - _ce(c.base, c.gamma)
    vol = float(c.nav.pct_change().std() * math.sqrt(periods_per_year(c.nav.index)))
    mdd = max_drawdown(c.nav)[0]
    problems = []
    if c.lab.profile is not None:
        from . import profiles
        problems = profiles.suitability(c.lab.profile, vol, mdd)
    status = FAIL if problems else _gap_status(gap)
    sentence = (f"Inadaptée au profil : {'; '.join(problems)}." if problems else
                f"Écart de {_pct(gap, 2, True)} par an avec la {c.ref.label}, après ajustement au risque.")
    return Verdict(*QUESTIONS[0], status, sentence,
                   {"écart ajusté": gap, "rendement": _cagr(c.nav), "rendement référence": _cagr(c.base),
                    "volatilité": vol, "pire chute": mdd})


def t2_covariance(c: Context, days: int = 126) -> Verdict:
    from .builder import _covariance
    held = c.strategy.weights.loc[c.nav.index[0]:]
    month_ends = held.resample("ME").last().index
    rets = c.panel.pct_change()
    rows = []
    for a, b in zip(month_ends[:-1], month_ends[1:]):
        if a not in c.panel.index:
            a = c.panel.index[c.panel.index.get_indexer([a], method="pad")[0]]
        w = held.loc[:a].iloc[-1]
        w = w[w.abs() > 1e-9]
        if w.empty:
            continue
        window = c.panel.loc[:a, list(w.index)].iloc[-(days + 1):]
        if len(window) <= days:
            continue
        cov = _covariance(window, days) * 252
        predicted = math.sqrt(max(float(w.values @ cov @ w.values), 0.0))
        nxt = rets.loc[a:b, list(w.index)].iloc[1:]
        if len(nxt) < 10 or predicted <= 0:
            continue
        realized = float((nxt.values @ w.values).std(ddof=1) * math.sqrt(252))
        rows.append({"mois": b.strftime("%Y-%m"), "risque prévu": predicted, "risque réalisé": realized})
    table = pd.DataFrame(rows)
    if len(table) < 24:
        return Verdict(*QUESTIONS[1], NA, "Trop peu de mois investis pour juger la prévision du risque.")
    corr = float(table["risque prévu"].corr(table["risque réalisé"]))
    bias = float((table["risque réalisé"] / table["risque prévu"]).median())
    status = PASS if corr >= 0.5 and 0.8 <= bias <= 1.25 else (BORDER if corr >= 0.3 else FAIL)
    under = " sous-estimé" if bias > 1.1 else (" surestimé" if bias < 0.9 else "")
    sentence = (f"Corrélation de {_n(corr)} entre le risque prévu et le risque du mois suivant ; "
                f"le risque réalisé vaut {_n(bias)} fois le risque prévu (médiane){', risque' + under if under else ''}.")
    return Verdict(*QUESTIONS[1], status, sentence,
                   {"corrélation": corr, "rapport réalisé / prévu": bias, "mois": len(table)}, table)


def t3_neutral(c: Context) -> Verdict:
    def neutral_factory(**params):
        inner = c.factory(**params)

        def fn(ctx):
            w = inner(ctx)
            if w is None:
                return None
            w = pd.Series(w, dtype="float64")
            picked = w[w > 1e-9]
            if picked.empty:
                return {}
            return {t: float(picked.sum() / len(picked)) for t in picked.index}
        return fn
    neutral = run_backtest(c.panel, neutral_factory(**c.spec.params), c.config, label="parts égales")
    nav_n = neutral.nav.loc[c.nav.index[0]:]
    gap = _ce(c.nav, c.gamma) - _ce(nav_n, c.gamma)
    if gap >= EQUALITY:
        status, s = PASS, f"Les pondérations ajoutent {_pct(gap, 2, True)} par an face à des parts égales sur les mêmes choix."
    elif gap > -EQUALITY:
        status, s = BORDER, (f"Les pondérations n'apportent rien de mesurable ({_pct(gap, 2, True)} par an) : "
                             "l'avantage vient du choix des actifs, pas de leurs poids.")
    else:
        status, s = FAIL, f"Des parts égales sur les mêmes choix auraient fait mieux ({_pct(-gap, 2)} par an)."
    return Verdict(*QUESTIONS[2], status, s, {"écart vs parts égales": gap,
                                              "rendement parts égales": _cagr(nav_n)})


def t4_leverage(c: Context) -> Verdict:
    ppy = periods_per_year(c.nav.index)
    rs, rb = c.nav.pct_change().dropna(), c.base.pct_change().dropna()
    vol_s, vol_b = float(rs.std() * math.sqrt(ppy)), float(rb.std() * math.sqrt(ppy))
    if vol_b <= 0:
        return Verdict(*QUESTIONS[3], NA, "La référence n'a pas de risque mesurable.")
    k = vol_s / vol_b
    cash = float(c.config.cash_rate) / ppy
    borrow = (float(c.config.cash_rate) + FINANCING_SPREAD) / ppy
    mix = k * rb + (1 - k) * (cash if k <= 1 else borrow)
    gap = certainty_equivalent(rs, c.gamma, ppy) - certainty_equivalent(mix, c.gamma, ppy)
    how = (f"la référence allégée à {_p0(k)} (le reste en liquidités)" if k <= 1
           else f"la référence renforcée à {_p0(k)} (emprunt au taux des liquidités + {_pct(FINANCING_SPREAD)})")
    status = _gap_status(gap)
    sentence = f"Au même risque, face à {how} : écart de {_pct(gap, 2, True)} par an après ajustement au risque."
    return Verdict(*QUESTIONS[3], status, sentence, {"exposition équivalente": k, "écart": gap,
                                                     "volatilité stratégie": vol_s, "volatilité référence": vol_b})


def t5_concentration(c: Context) -> Verdict:
    ms = c.nav.resample("ME").last().pct_change().dropna()
    mb = c.base.resample("ME").last().pct_change().dropna()
    excess = (ms - mb).dropna()
    if len(excess) < 24:
        return Verdict(*QUESTIONS[4], NA, "Moins de deux ans de mois comparables.")
    total = float(excess.sum())
    k = max(1, math.ceil(0.05 * len(excess)))
    without = float(excess.sort_values().iloc[:-k].sum())
    yearly = (1 + ms).groupby(ms.index.year).prod() - (1 + mb).groupby(mb.index.year).prod()
    share_years = float((yearly > 0).mean())
    best = excess.idxmax()
    if total <= 0:
        status, s = NA, "Pas de gain cumulé à répartir : la stratégie ne fait pas mieux que sa référence."
    elif without > 0 and share_years >= 0.5:
        status, s = PASS, (f"Sans ses {k} meilleurs mois, il reste {_p0(without / total)} du gain ; "
                           f"elle fait mieux que la référence {_p0(share_years)} des années.")
    elif without > 0:
        status, s = BORDER, (f"Le gain survit sans ses {k} meilleurs mois, mais seulement {_p0(share_years)} "
                             "des années sont gagnantes.")
    else:
        status, s = FAIL, (f"Tout le gain tient dans {k} mois (le meilleur : {best:%m/%Y}) : sans eux, elle fait "
                           f"moins bien que la référence. Elle ne gagne que {_p0(share_years)} des années.")
    return Verdict(*QUESTIONS[4], status, s, {"gain mensuel cumulé": total, "sans les meilleurs mois": without,
                                              "mois retirés": k, "années gagnantes": share_years})


def t6_montecarlo(c: Context, n_paths: int = 2000) -> Verdict:
    from .robustness import block_bootstrap
    table = block_bootstrap(c.nav, c.base, n_paths=n_paths, ref=c.ref)
    row = table[table["mesure"] == c.ref.ce_col].iloc[0]
    p = float(row["probabilité > 0"])
    status = PASS if p >= 0.6 else (BORDER if p >= 0.5 else FAIL)
    sentence = (f"Sur {n_paths} histoires rebattues par blocs de 6 mois, elle fait mieux que la référence "
                f"dans {_p0(p)} des cas (écart médian {_pct(float(row['médiane']), 2, True)} par an ; "
                f"5 % des cas sous {_pct(float(row['5 %']), 2, True)}).")
    return Verdict(*QUESTIONS[5], status, sentence, {"probabilité de faire mieux": p,
                                                     "écart médian": float(row["médiane"]),
                                                     "écart 5 %": float(row["5 %"])}, table)


def inflation() -> pd.Series:
    frame = pd.read_csv(INFLATION_FILE, comment="#")
    s = pd.Series(frame["indice"].astype(float).values,
                  index=pd.to_datetime(frame["mois"]) + pd.offsets.MonthEnd(0), name="IPCH France")
    return s


def t7_purchasing_power(c: Context) -> Verdict:
    cpi = inflation()
    ms = c.nav.resample("ME").last()
    mb = c.base.resample("ME").last()
    common = ms.index.intersection(cpi.index)
    if len(common) < 24:
        return Verdict(*QUESTIONS[6], NA, "Pas assez de mois couverts par l'indice des prix.")
    ms, mb, p = ms.loc[common], mb.loc[common], cpi.loc[common]
    yrs = (common[-1] - common[0]).days / 365.25
    real_s = float(((ms.iloc[-1] / ms.iloc[0]) / (p.iloc[-1] / p.iloc[0])) ** (1 / yrs) - 1)
    real_b = float(((mb.iloc[-1] / mb.iloc[0]) / (p.iloc[-1] / p.iloc[0])) ** (1 / yrs) - 1)
    infl = float((p.iloc[-1] / p.iloc[0]) ** (1 / yrs) - 1)
    yoy = cpi.pct_change(12).reindex(common)
    excess = (ms.pct_change() - mb.pct_change()).dropna()
    hot = excess[yoy.reindex(excess.index) >= 0.03]
    cash_share = float((1 - c.strategy.weights.loc[c.nav.index[0]:].sum(axis=1)).clip(lower=0).mean())
    if real_s <= 0:
        status = FAIL
    elif real_s > real_b:
        status = PASS
    else:
        status = BORDER
    hot_text = (f" Quand l'inflation dépasse 3 % sur un an ({len(hot)} mois), elle fait "
                f"{_pct(float(hot.mean()) * 12, 1, True)} par an face à la référence." if len(hot) >= 6 else "")
    sentence = (f"Après inflation ({_pct(infl)} par an en France), elle rapporte {_pct(real_s, 1, True)} par an, "
                f"contre {_pct(real_b, 1, True)} pour la référence.{hot_text} Liquidités moyennes : {_p0(cash_share)}.")
    return Verdict(*QUESTIONS[6], status, sentence,
                   {"rendement réel": real_s, "rendement réel référence": real_b, "inflation": infl,
                    "mois d'inflation forte": len(hot), "liquidités moyennes": cash_share,
                    "fin de l'indice": str(cpi.index[-1].date())})


def t8_reverse_stress(c: Context) -> Verdict:
    definition = c.spec.params.get("definition") if c.spec.rule == "composite" else None
    classes = ({u["ticker"]: u["classe"] for u in definition["univers"]} if definition else {})
    equities = [t for t in c.panel.columns if classes.get(t) == "actions"] or [c.panel.columns[0]]
    rets = c.panel.pct_change().dropna()
    market = rets[equities].mean(axis=1)
    betas = {t: float(np.cov(rets[t], market)[0, 1] / market.var()) for t in c.panel.columns}
    w = c.strategy.weights.iloc[-1]
    w = w[w.abs() > 1e-9]
    beta_p = float(sum(w[t] * betas.get(t, 0.0) for t in w.index))
    limit = c.lab.profile.max_loss if c.lab.profile is not None else 0.25
    who = (f"la limite du profil {c.lab.profile.label} (−{_p0(limit)})" if c.lab.profile is not None
           else f"une perte de {_p0(limit)}")
    market_nav = (1 + market).cumprod()
    worst_market = max_drawdown(market_nav)[0]
    if beta_p <= 0.02:
        return Verdict(*QUESTIONS[7], PASS, "Le portefeuille actuel ne dépend presque pas des actions : "
                       f"aucune baisse des actions ne l'amène à {who}.", {"bêta actions": beta_p})
    breaking = limit / beta_p
    if breaking > 1:
        status, s = PASS, f"Même une perte totale des actions ne suffirait pas à atteindre {who} (bêta {_n(beta_p)})."
    else:
        status = PASS if breaking >= abs(worst_market) else (BORDER if breaking >= 0.6 * abs(worst_market) else FAIL)
        s = (f"Avec les poids actuels (bêta actions {_n(beta_p)}), {who} est atteinte si les actions perdent "
             f"{_p0(breaking)} d'un coup. Pire baisse des actions sur la période : {_pct(worst_market, 0)}.")
    return Verdict(*QUESTIONS[7], status, s, {"bêta actions": beta_p, "baisse de rupture": min(breaking, 1.0),
                                              "pire baisse des actions": worst_market})


def t9_triage(c: Context) -> Verdict:
    from .analytics import sharpe_statistics
    stats = sharpe_statistics(c.nav)
    n_dec = int(len(c.strategy.decisions))
    need, have = stats["historique minimal nécessaire (années)"], stats["historique disponible (années)"]
    if n_dec < 30:
        status, s = FAIL, f"{n_dec} décisions seulement : moins de 30, on ne peut rien conclure."
    elif have >= need:
        status, s = PASS, (f"{n_dec} décisions sur {_n(have, 1)} ans ; il en faudrait {_n(need, 1)} pour que son "
                           "Sharpe soit significatif à 95 % : l'échantillon suffit.")
    else:
        status, s = BORDER, (f"{n_dec} décisions sur {_n(have, 1)} ans, mais il faudrait "
                             f"{'plus de 100' if not math.isfinite(need) or need > 100 else _n(need, 1)} ans "
                             "pour que son Sharpe soit significatif à 95 %.")
    return Verdict(*QUESTIONS[8], status, s, {"décisions": n_dec, "années disponibles": have,
                                              "années nécessaires": need})


def t10_deflated(c: Context) -> Verdict:
    from .analytics import sharpe_statistics
    sharpes = [e["is"]["sharpe"] for e in c.lab._events("trial") if e["is"].get("sharpe") is not None]
    n = max(c.lab.n_trials, 2)
    stats = sharpe_statistics(c.nav, n_trials=n, trial_sharpes=sharpes if len(sharpes) > 1 else None)
    p = float(stats["Sharpe dégonflé (DSR) - probabilité réelle"])
    status = PASS if p >= 0.9 else (BORDER if p >= 0.8 else FAIL)
    s = (f"Avec {n} essais au registre, la probabilité que son Sharpe ({_n(stats['Sharpe annualisé'])}) soit réel "
         f"est de {_p0(p)} ; par pur hasard, le meilleur de {n} essais atteindrait déjà "
         f"{_n(stats['Sharpe attendu par chance au meilleur des essais'])}.")
    return Verdict(*QUESTIONS[9], status, s, {"probabilité": p, "essais": n,
                                                               "Sharpe": stats["Sharpe annualisé"]})


def t11_oos(c: Context) -> Verdict:
    lab, spec = c.lab, c.spec
    fp = lab.fingerprint(spec)
    tests = [e for e in lab._events("oos") if e["fingerprint"] == fp and e.get("profile") == lab.profile_name]
    if not tests:
        frozen = any(e["fingerprint"] == fp for e in lab._events("freeze"))
        return Verdict(*QUESTIONS[10], NA, "Période scellée non consultée : " + (
            "la stratégie est gelée, son test unique reste à faire." if frozen else
            "la stratégie n'est pas gelée. Cette batterie ne l'ouvre jamais."))
    oos = tests[0]["oos"]
    trial = next((e for e in lab._events("trial") if e["fingerprint"] == fp
                  and e.get("profile") == lab.profile_name), None)
    is_gap = trial["is"]["ce3_gap_vs_1N"] if trial else None
    oos_gap = oos["ce3_gap_vs_1N"]
    if is_gap is None or is_gap <= 0:
        status = PASS if oos["verdict"] == "ADDS_VALUE" else BORDER
        s = f"Test scellé : {_pct(oos_gap, 2, True)} par an face à la référence ({oos['verdict']})."
        return Verdict(*QUESTIONS[10], status, s, {"écart test": oos_gap})
    keep = oos_gap / is_gap
    status = PASS if oos["verdict"] == "ADDS_VALUE" and keep >= 0.5 else (
        BORDER if oos["verdict"] in ("ADDS_VALUE", "NO_REAL_DIFFERENCE") else FAIL)
    kept = (f"elle a fait mieux en test qu'en construction ({_n(keep, 1)} fois l'avantage)" if keep > 1
            else f"elle a gardé {_p0(keep)} de son avantage")
    s = (f"Construction : {_pct(is_gap, 2, True)} par an ; test scellé : {_pct(oos_gap, 2, True)} : "
         f"{kept} ({oos['verdict']}, testée le {pd.Timestamp(tests[0]['at'][:10]):%d/%m/%Y}).")
    return Verdict(*QUESTIONS[10], status, s, {"écart construction": is_gap, "écart test": oos_gap,
                                               "part conservée": keep})


def t12_multi_asset(c: Context, pool: pd.DataFrame | None, pool_classes: dict[str, str] | None,
                    n_baskets: int = 40, seed: int = 0, progress: Callable | None = None) -> Verdict:
    if c.spec.rule != "composite" or pool is None or pool.empty:
        return Verdict(*QUESTIONS[11], NA, "Réservé aux stratégies construites par blocs, avec un réservoir d'actifs.")
    import copy
    from . import profiles
    from .builder import composite
    from .strategies import equal_weight
    definition = c.spec.params["definition"]
    own = {u["ticker"] for u in definition["univers"]}
    candidates = [t for t in pool.columns if t not in own and pool_classes.get(t)]
    by_class: dict[str, list[str]] = {}
    for t in candidates:
        by_class.setdefault(pool_classes[t], []).append(t)
    need: dict[str, int] = {}
    for u in definition["univers"]:
        need[u["classe"]] = need.get(u["classe"], 0) + 1
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_baskets):
        basket = []
        for cls, k in need.items():
            avail = [t for t in by_class.get(cls, []) if t not in basket]
            if avail:
                basket += list(rng.choice(avail, size=min(k, len(avail)), replace=False))
        if len(basket) < 3:
            continue
        prices = pool[basket].loc[pool.index < c.lab.oos_start].dropna()
        if len(prices) < 750:
            continue
        defn = copy.deepcopy(definition)
        defn["univers"] = [{"ticker": t, "classe": pool_classes[t]} for t in basket]
        for blk in defn.get("selection", []):                    # exclusions of absent tickers
            if blk.get("type") == "exclude":
                blk["tickers"] = [t for t in blk.get("tickers", []) if t in basket]
        risk = defn.get("risque", {})
        if risk.get("refuge") and risk["refuge"] not in basket:
            bonds = [t for t in basket if pool_classes[t] == "obligations"]
            if bonds:
                risk["refuge"] = bonds[0]
            else:
                risk.pop("refuge")
        try:
            rule = composite(defn)
            ref_rule = (profiles.reference_rule(c.lab.profile, defn["univers"])()
                        if c.lab.profile is not None else equal_weight())
            s = run_backtest(prices, rule, c.config)
            b = run_backtest(prices, ref_rule, c.config)
        except Exception:
            continue
        start = max(_first_exec(s), _first_exec(b))
        gap = _ce(s.nav.loc[start:], c.gamma) - _ce(b.nav.loc[start:], c.gamma)
        rows.append({"panier": " · ".join(basket), "début": str(prices.index[0].date()), "écart": gap})
        if progress:
            progress((i + 1) / n_baskets)
    table = pd.DataFrame(rows)
    if len(table) < 10:
        return Verdict(*QUESTIONS[11], NA, "Pas assez d'actifs de mêmes classes hors de la stratégie pour tirer des paniers.",
                       {"paniers": len(table)}, table if len(table) else None)
    share = float((table["écart"] > 0).mean())
    status = PASS if share >= 0.6 else (BORDER if share >= 0.5 else FAIL)
    s = (f"Sur {len(table)} paniers tirés au hasard parmi {len(candidates)} actifs qu'elle n'a jamais utilisés "
         f"(mêmes classes d'actifs), elle fait mieux que la référence du panier dans {_p0(share)} des cas "
         f"(écart médian {_pct(float(table['écart'].median()), 2, True)} par an). Certains de ces actifs "
         "suivent les mêmes indices sous une autre cotation : le test est moins indépendant qu'il n'y paraît.")
    return Verdict(*QUESTIONS[11], status, s, {"paniers": len(table), "part gagnante": share,
                                               "écart médian": float(table["écart"].median())}, table)


# =============================================================================== all of them
def battery(lab, spec, *, pool: pd.DataFrame | None = None, pool_classes: dict | None = None,
            n_baskets: int = 40, progress: Callable[[float, str], None] | None = None) -> list[Verdict]:
    """The twelve verdicts, in order. A test that cannot run says why instead of failing."""
    c = context(lab, spec)
    steps = [t1_reference, t2_covariance, t3_neutral, t4_leverage, t5_concentration, t6_montecarlo,
             t7_purchasing_power, t8_reverse_stress, t9_triage, t10_deflated, t11_oos]
    out = []
    for i, test in enumerate(steps):
        if progress:
            progress(i / 12, QUESTIONS[i][1])
        try:
            out.append(test(c))
        except Exception as exc:                      # one broken test must not hide the others
            out.append(Verdict(*QUESTIONS[i], NA, f"Calcul impossible : {exc}"))
    if progress:
        progress(11 / 12, QUESTIONS[11][1])
    try:
        out.append(t12_multi_asset(c, pool, pool_classes or {}, n_baskets,
                                   progress=(lambda f: progress((11 + f) / 12, QUESTIONS[11][1])) if progress else None))
    except Exception as exc:
        out.append(Verdict(*QUESTIONS[11], NA, f"Calcul impossible : {exc}"))
    return out


def summary(verdicts: list[Verdict]) -> dict:
    counts = {s: sum(v.status == s for v in verdicts) for s in (PASS, BORDER, FAIL, NA)}
    return {"passés": counts[PASS], "limites": counts[BORDER], "échoués": counts[FAIL],
            "non applicables": counts[NA], "jugés": 12 - counts[NA]}
