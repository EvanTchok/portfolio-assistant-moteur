"""Full performance and risk analysis of a backtest, against a benchmark.

Every figure is computed from the NAV curve (and the weights for composition). The
functions are grouped the way a professional tear sheet is read:

    returns        calendar tables, best/worst, hit rate, rolling returns
    risk           volatility, downside, skew/kurtosis, VaR/CVaR (3 methods), drawdowns
    ratios         Sharpe, Sortino, Calmar, Omega, gain-to-pain, tail ratio, Ulcer
    relative       alpha, beta, correlation, tracking error, information ratio, captures
    statistics     t-stat, probabilistic Sharpe, minimum track record, deflated Sharpe
    composition    average weights, turnover, concentration, return and risk contribution
    stress         named crises and up/down market behaviour

No scipy: the normal distribution comes from the standard library.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from .engine import BacktestResult
from .metrics import max_drawdown, periods_per_year

N01 = NormalDist()
EULER_GAMMA = 0.5772156649015329

# Named market stress windows (peak -> trough of the S&P 500 or the relevant market).
CRISES = {
    "Crise financière 2008": ("2007-10-09", "2009-03-09"),
    "Crise dette euro 2011": ("2011-07-07", "2011-10-03"),
    "Chine / pétrole 2015-16": ("2015-08-17", "2016-02-11"),
    "Krach T4 2018": ("2018-09-20", "2018-12-24"),
    "Covid 2020": ("2020-02-19", "2020-03-23"),
    "Inflation / taux 2022": ("2022-01-03", "2022-10-12"),
    "Droits de douane 2025": ("2025-02-19", "2025-04-08"),
}


# ------------------------------------------------------------------------- helpers
def _rets(nav: pd.Series) -> pd.Series:
    return nav.pct_change().dropna()


def _monthly(nav: pd.Series) -> pd.Series:
    month_end = nav.resample("ME").last()
    first = pd.Series([nav.iloc[0]], index=[nav.index[0] - pd.Timedelta(days=1)])
    return pd.concat([first, month_end]).pct_change().dropna()


def _cagr(nav: pd.Series) -> float:
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    return float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1) if years > 0 else float("nan")


def _align(nav: pd.Series, bench: pd.Series) -> tuple[pd.Series, pd.Series]:
    if not nav.index.equals(bench.index):
        raise ValueError("strategy and benchmark must share the same dates")
    return nav, bench


# ------------------------------------------------------------------------- returns
def calendar_returns(nav: pd.Series, bench: pd.Series | None = None) -> pd.DataFrame:
    """One row per year: strategy, benchmark, difference. Partial years are flagged."""
    rows = []
    for year, chunk in nav.groupby(nav.index.year):
        prev = nav.loc[:chunk.index[0]].iloc[-2] if chunk.index[0] != nav.index[0] else chunk.iloc[0]
        row = {"année": year, "stratégie": chunk.iloc[-1] / prev - 1,
               "partielle": bool(chunk.index[0].month > 1 or chunk.index[-1].month < 12)}
        if bench is not None:
            b = bench.loc[chunk.index]
            b_prev = bench.loc[:chunk.index[0]].iloc[-2] if chunk.index[0] != bench.index[0] else b.iloc[0]
            row["référence"] = b.iloc[-1] / b_prev - 1
            row["écart"] = row["stratégie"] - row["référence"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("année")


def monthly_table(nav: pd.Series) -> pd.DataFrame:
    """Year x month grid of monthly returns, plus the year total."""
    m = _monthly(nav)
    grid = pd.DataFrame({"année": m.index.year, "mois": m.index.month, "r": m.values})
    table = grid.pivot(index="année", columns="mois", values="r")
    names = ["jan", "fév", "mar", "avr", "mai", "juin", "juil", "août", "sep", "oct", "nov", "déc"]
    table = table.reindex(columns=range(1, 13))
    table.columns = names
    table["total année"] = (1 + table.fillna(0)).prod(axis=1) - 1
    return table


def return_stats(nav: pd.Series) -> dict:
    r, m = _rets(nav), _monthly(nav)
    yearly = nav.resample("YE").last().pct_change().dropna()
    rolling_12m = nav.pct_change(int(round(periods_per_year(nav.index)))).dropna()
    return {
        "rendement total": float(nav.iloc[-1] / nav.iloc[0] - 1),
        "rendement annualisé (CAGR)": _cagr(nav),
        "meilleur jour": float(r.max()), "pire jour": float(r.min()),
        "meilleur mois": float(m.max()), "pire mois": float(m.min()),
        "meilleure année": float(yearly.max()) if len(yearly) else float("nan"),
        "pire année": float(yearly.min()) if len(yearly) else float("nan"),
        "% de mois positifs": float((m > 0).mean()),
        "gain moyen mois positif": float(m[m > 0].mean()) if (m > 0).any() else float("nan"),
        "perte moyenne mois négatif": float(m[m < 0].mean()) if (m < 0).any() else float("nan"),
        "rendement 12 mois glissant - min": float(rolling_12m.min()) if len(rolling_12m) else float("nan"),
        "rendement 12 mois glissant - médiane": float(rolling_12m.median()) if len(rolling_12m) else float("nan"),
        "rendement 12 mois glissant - max": float(rolling_12m.max()) if len(rolling_12m) else float("nan"),
    }


# ---------------------------------------------------------------------------- risk
def value_at_risk(r: pd.Series, level: float = 0.95) -> dict:
    """Daily VaR and CVaR, as positive losses, by three methods."""
    x = np.asarray(r, dtype=float)
    q = 1 - level
    hist_var = -float(np.quantile(x, q))
    tail = x[x <= -hist_var]
    mu, sd = float(x.mean()), float(x.std(ddof=1))
    z = N01.inv_cdf(q)
    s = float(pd.Series(x).skew())
    k = float(pd.Series(x).kurt())  # excess kurtosis
    z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
    return {
        "VaR historique": hist_var,
        "VaR gaussienne": -(mu + z * sd),
        "VaR Cornish-Fisher": -(mu + z_cf * sd),
        "CVaR historique (perte moyenne au-delà)": -float(tail.mean()) if tail.size else float("nan"),
    }


def drawdown_table(nav: pd.Series, top: int = 5) -> pd.DataFrame:
    """The deepest drawdown episodes: peak, trough, recovery, depth and durations (days)."""
    values = nav.to_numpy()
    peak_val, peak_i, episodes, i = values[0], 0, [], 1
    current = None
    for i in range(1, len(values)):
        if values[i] >= peak_val:
            if current is not None:
                current["recovery"] = i
                episodes.append(current)
                current = None
            peak_val, peak_i = values[i], i
        else:
            depth = values[i] / peak_val - 1
            if current is None:
                current = {"peak": peak_i, "trough": i, "depth": depth, "recovery": None}
            elif depth < current["depth"]:
                current["trough"], current["depth"] = i, depth
    if current is not None:
        episodes.append(current)
    episodes = sorted(episodes, key=lambda e: e["depth"])[:top]
    idx = nav.index
    rows = []
    for e in episodes:
        rec = e["recovery"]
        rows.append({
            "sommet": idx[e["peak"]].date(), "creux": idx[e["trough"]].date(),
            "retour au sommet": idx[rec].date() if rec is not None else "pas encore",
            "profondeur": e["depth"],
            "jours de baisse": (idx[e["trough"]] - idx[e["peak"]]).days,
            "jours pour récupérer": (idx[rec] - idx[e["trough"]]).days if rec is not None else None,
            "durée totale (jours)": ((idx[rec] if rec is not None else idx[-1]) - idx[e["peak"]]).days,
        })
    return pd.DataFrame(rows)


def risk_stats(nav: pd.Series) -> dict:
    r = _rets(nav)
    ppy = periods_per_year(nav.index)
    mdd, _ = max_drawdown(nav)
    dd = nav / nav.cummax() - 1
    out = {
        "volatilité annualisée": float(r.std(ddof=1) * math.sqrt(ppy)),
        "volatilité baissière (downside)": float(np.sqrt(np.mean(np.minimum(r, 0) ** 2)) * math.sqrt(ppy)),
        "asymétrie (skewness)": float(r.skew()),
        "aplatissement (excess kurtosis)": float(r.kurt()),
        "perte maximale (max drawdown)": mdd,
        "drawdown actuel": float(dd.iloc[-1]),
        "drawdown moyen": float(dd.mean()),
        "indice Ulcer": float(np.sqrt(np.mean((dd * 100) ** 2))),
    }
    # the longest underwater spell measured in calendar days, not bars
    under_mask = (dd < -1e-12).to_numpy()
    longest, start = 0, None
    for i, flag in enumerate(under_mask):
        if flag and start is None:
            start = i
        if (not flag or i == len(under_mask) - 1) and start is not None:
            end = i if flag else i - 1
            longest = max(longest, (nav.index[end] - nav.index[start]).days)
            start = None
    out["plus longue période sous le sommet (jours)"] = longest
    for level in (0.95, 0.99):
        for name, value in value_at_risk(r, level).items():
            out[f"{name} {level:.0%} (1 jour)"] = value
    return out


# -------------------------------------------------------------------------- ratios
def ratio_stats(nav: pd.Series, rf: float = 0.0) -> dict:
    r = _rets(nav)
    ppy = periods_per_year(nav.index)
    rf_p = (1 + rf) ** (1 / ppy) - 1
    ex = r - rf_p
    vol = r.std(ddof=1) * math.sqrt(ppy)
    down = np.sqrt(np.mean(np.minimum(ex, 0) ** 2)) * math.sqrt(ppy)
    cagr = _cagr(nav)
    mdd, _ = max_drawdown(nav)
    m = _monthly(nav)
    dd = nav / nav.cummax() - 1
    ulcer = float(np.sqrt(np.mean((dd * 100) ** 2)))
    gains, losses = ex[ex > 0].sum(), -ex[ex < 0].sum()
    return {
        "Sharpe": float(ex.mean() / r.std(ddof=1) * math.sqrt(ppy)) if vol > 0 else float("nan"),
        "Sortino": float(ex.mean() * ppy / down) if down > 0 else float("inf"),
        "Calmar (CAGR / perte max)": cagr / abs(mdd) if mdd < 0 else float("inf"),
        "Omega (seuil = taux sans risque)": float(gains / losses) if losses > 0 else float("inf"),
        "Gain / douleur (mensuel)": float(m[m > 0].sum() / -m[m < 0].sum()) if (m < 0).any() else float("inf"),
        "Ratio de queue (95e / 5e centile)": float(abs(np.quantile(r, 0.95) / np.quantile(r, 0.05))),
        "Ulcer performance index": (cagr - rf) / (ulcer / 100) if ulcer > 0 else float("inf"),
    }


# ------------------------------------------------------------------------ relative
def relative_stats(nav: pd.Series, bench: pd.Series, rf: float = 0.0) -> dict:
    nav, bench = _align(nav, bench)
    r, b = _rets(nav), _rets(bench)
    ppy = periods_per_year(nav.index)
    rf_p = (1 + rf) ** (1 / ppy) - 1
    xs, xb = r - rf_p, b - rf_p
    beta = float(np.cov(xs, xb, ddof=1)[0, 1] / np.var(xb, ddof=1))
    alpha_p = float(xs.mean() - beta * xb.mean())
    corr = float(np.corrcoef(r, b)[0, 1])
    active = r - b
    te = float(active.std(ddof=1) * math.sqrt(ppy))
    ms, mb = _monthly(nav), _monthly(bench)
    up, down = mb > 0, mb < 0
    vol_s, vol_b = r.std(ddof=1) * math.sqrt(ppy), b.std(ddof=1) * math.sqrt(ppy)
    sharpe_s = xs.mean() / r.std(ddof=1) * math.sqrt(ppy)
    return {
        "bêta": beta,
        "alpha annualisé (Jensen)": float((1 + alpha_p) ** ppy - 1),
        "corrélation": corr,
        "R²": corr ** 2,
        "écart de rendement annuel (CAGR)": _cagr(nav) - _cagr(bench),
        "tracking error": te,
        "information ratio": float(active.mean() * ppy / te) if te > 0 else float("nan"),
        "capture à la hausse": float(ms[up].mean() / mb[up].mean()) if up.any() else float("nan"),
        "capture à la baisse": float(ms[down].mean() / mb[down].mean()) if down.any() else float("nan"),
        "% de mois meilleurs que la référence": float((ms > mb).mean()),
        "Treynor": float(xs.mean() * ppy / beta) if beta else float("nan"),
        "M² (Sharpe ramené à la vol. de la référence)": float(sharpe_s * vol_b + rf),
        "volatilité relative (stratégie / référence)": float(vol_s / vol_b),
    }


# ---------------------------------------------------------------------- statistics
def sharpe_statistics(nav: pd.Series, *, n_trials: int | None = None,
                      trial_sharpes: list[float] | None = None, confidence: float = 0.95) -> dict:
    """Is the Sharpe ratio real or luck? (Bailey & López de Prado)

    Works in per-period units; reports annualised figures for reading.
    """
    r = _rets(nav).to_numpy()
    n = r.size
    ppy = periods_per_year(nav.index)
    sr = float(r.mean() / r.std(ddof=1))
    skew = float(pd.Series(r).skew())
    kurt = float(pd.Series(r).kurt()) + 3.0
    var_term = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    se = math.sqrt(var_term / (n - 1))
    out = {
        "Sharpe annualisé": sr * math.sqrt(ppy),
        "t-stat du rendement moyen": float(r.mean() / (r.std(ddof=1) / math.sqrt(n))),
        "Sharpe - intervalle 95 % bas": (sr - 1.96 * se) * math.sqrt(ppy),
        "Sharpe - intervalle 95 % haut": (sr + 1.96 * se) * math.sqrt(ppy),
        "probabilité que le vrai Sharpe > 0 (PSR)": N01.cdf(sr / se),
    }
    z = N01.inv_cdf(confidence)
    out["historique minimal nécessaire (années)"] = (
        float((1 + var_term * (z / sr) ** 2) / ppy) if sr > 0 else float("inf"))
    out["historique disponible (années)"] = n / ppy
    if n_trials and n_trials > 1:
        if trial_sharpes and len(trial_sharpes) > 1:
            v = float(np.var(np.asarray(trial_sharpes) / math.sqrt(ppy), ddof=1))
        else:
            v = var_term / (n - 1)  # fallback: estimation noise of one Sharpe
        sr0 = math.sqrt(v) * ((1 - EULER_GAMMA) * N01.inv_cdf(1 - 1 / n_trials)
                              + EULER_GAMMA * N01.inv_cdf(1 - 1 / (n_trials * math.e)))
        out["essais pris en compte"] = int(n_trials)
        out["Sharpe attendu par chance au meilleur des essais"] = sr0 * math.sqrt(ppy)
        out["Sharpe dégonflé (DSR) - probabilité réelle"] = N01.cdf((sr - sr0) / se)
    return out


# --------------------------------------------------------------------- composition
def composition(result: BacktestResult, start=None) -> pd.DataFrame:
    """Per asset: average weight, share of return, share of risk."""
    w = result.weights.loc[start:] if start is not None else result.weights
    if result.prices is None:
        raise ValueError("this result carries no prices: rerun it with the current engine")
    asset_r = result.prices.pct_change().loc[w.index]
    w_prev = result.weights.shift(1).loc[w.index]
    contrib = (w_prev * asset_r).sum()
    total = contrib.sum()
    cov = asset_r.dropna(how="all").cov()
    w_avg = w.mean()
    sigma_w = cov.fillna(0).to_numpy() @ w_avg.to_numpy()
    port_var = float(w_avg.to_numpy() @ sigma_w)
    risk = w_avg.to_numpy() * sigma_w / port_var if port_var > 0 else np.full(len(w_avg), np.nan)
    return pd.DataFrame({
        "poids moyen": w_avg,
        "poids max": w.max(),
        "% du temps détenu": (w > 1e-6).mean(),
        "contribution au rendement": contrib,
        "part du rendement": contrib / total if total else np.nan,
        "part du risque": risk,
    })


def portfolio_stats(result: BacktestResult, start=None) -> dict:
    w = result.weights.loc[start:] if start is not None else result.weights
    invested = w.sum(axis=1)
    hhi = (w.pow(2).sum(axis=1)).replace(0, np.nan)
    trades = result.trades if start is None else result.trades[result.trades["date"] >= start]
    years = max((w.index[-1] - w.index[0]).days / 365.25, 1e-9)
    nav = result.nav.loc[w.index]
    traded = trades.groupby("date")["notional"].apply(lambda s: s.abs().sum()) if len(trades) else pd.Series(dtype=float)
    turnover = float((traded / nav.reindex(traded.index)).sum() / 2 / years) if len(traded) else 0.0
    return {
        "part investie moyenne": float(invested.mean()),
        "part en cash moyenne": float(1 - invested.mean()),
        "nombre effectif de lignes (1/HHI) moyen": float((invested**2 / hhi).mean()),
        "poids maximal atteint par une ligne": float(w.max().max()),
        "rotation annuelle (turnover)": turnover,
        "frais payés": float(trades["fee"].sum()) if len(trades) else 0.0,
        "frais en % du capital moyen par an": float(trades["fee"].sum() / nav.mean() / years) if len(trades) else 0.0,
        "nombre de rebalancements": int((result.decisions["execution_date"] >= w.index[0]).sum())
        if not result.decisions.empty else 0,
    }


# -------------------------------------------------------------------------- stress
def crisis_table(nav: pd.Series, bench: pd.Series) -> pd.DataFrame:
    nav, bench = _align(nav, bench)
    rows = []
    for name, (a, b) in CRISES.items():
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        if a < nav.index[0] or b > nav.index[-1]:
            continue
        s, r = nav.loc[a:b], bench.loc[a:b]
        if len(s) < 2:
            continue
        rows.append({"épisode": name, "du": a.date(), "au": b.date(),
                     "stratégie": s.iloc[-1] / s.iloc[0] - 1,
                     "référence": r.iloc[-1] / r.iloc[0] - 1,
                     "écart": (s.iloc[-1] / s.iloc[0]) - (r.iloc[-1] / r.iloc[0]),
                     "pire chute stratégie": max_drawdown(s)[0]})
    return pd.DataFrame(rows)


def market_regimes(nav: pd.Series, bench: pd.Series) -> pd.DataFrame:
    """Average monthly return when the benchmark rises vs when it falls."""
    ms, mb = _monthly(nav), _monthly(bench)
    rows = []
    for label, mask in (("mois où la référence monte", mb > 0), ("mois où la référence baisse", mb <= 0)):
        rows.append({"régime": label, "nombre de mois": int(mask.sum()),
                     "stratégie (moyenne)": float(ms[mask].mean()) if mask.any() else float("nan"),
                     "référence (moyenne)": float(mb[mask].mean()) if mask.any() else float("nan")})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- all
def full_analysis(strategy: BacktestResult, benchmark: BacktestResult, *, start=None,
                  rf: float = 0.0, n_trials: int | None = None,
                  trial_sharpes: list[float] | None = None) -> dict:
    """Every table, measured from ``start`` (default: first day both are invested)."""
    starts = [r.decisions["execution_date"].iloc[0] for r in (strategy, benchmark)
              if not r.decisions.empty]
    auto = max(starts) if starts else strategy.nav.index[0]
    start = max(auto, pd.Timestamp(start)) if start is not None else auto
    nav, bench = strategy.nav.loc[start:], benchmark.nav.loc[start:]

    def side_by_side(fn, **kw) -> pd.DataFrame:
        a, b = fn(nav, **kw), fn(bench, **kw)
        return pd.DataFrame({strategy.label: a, benchmark.label: b})

    return {
        "période": {"début": nav.index[0].date(), "fin": nav.index[-1].date(),
                    "années": (nav.index[-1] - nav.index[0]).days / 365.25,
                    "stratégie": strategy.label, "référence": benchmark.label},
        "rendement": side_by_side(return_stats),
        "risque": side_by_side(risk_stats),
        "ratios": side_by_side(ratio_stats, rf=rf),
        "relatif": pd.Series(relative_stats(nav, bench, rf=rf), name="valeur").to_frame(),
        "statistiques": pd.Series(sharpe_statistics(nav, n_trials=n_trials, trial_sharpes=trial_sharpes),
                                  name="valeur").to_frame(),
        "années": calendar_returns(nav, bench),
        "mois": monthly_table(nav),
        "drawdowns": drawdown_table(nav),
        "drawdowns référence": drawdown_table(bench),
        "crises": crisis_table(nav, bench),
        "régimes": market_regimes(nav, bench),
        "composition": composition(strategy, start),
        "portefeuille": pd.Series(portfolio_stats(strategy, start), name="valeur").to_frame(),
        "courbes": pd.DataFrame({strategy.label: nav / nav.iloc[0],
                                 benchmark.label: bench / bench.iloc[0]}),
    }
