"""Is the result robust, or does it depend on one lucky choice?

A professional reviewer asks the same questions of every strategy:

    parameters     does it still work with 9 months instead of 12? (or only with 12?)
    costs          at what transaction cost does the edge disappear?
    rebalancing    weekly, monthly, quarterly: does the answer change?
    entry date     would an investor who started in a different year agree?
    rolling        over every 3-year window, how often does it beat the reference?
    bootstrap      if history had been reshuffled, how wide is the range of outcomes?

Each function returns a table. All of them compare to a REFERENCE under identical
conditions: 1/N judged by a γ=3 investor by default, or a client profile's reference
portfolio judged with that profile's risk aversion (see profiles.py).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
import pandas as pd

from .engine import BacktestConfig, run_backtest
from .metrics import certainty_equivalent, max_drawdown, periods_per_year
from .builder import apply_settings, current_settings
from .strategies import equal_weight


@dataclass(frozen=True)
class Reference:
    """What a strategy is judged against, and by whom."""
    rule: Callable[[], Callable] = field(default=equal_weight)   # factory of the reference rule
    gamma: float = 3.0                                           # the judge's risk aversion
    label: str = "1/N"

    @property
    def ce_col(self) -> str:
        return f"écart CE γ={self.gamma:g} vs {self.label}"

    @property
    def cagr_col(self) -> str:
        return f"écart CAGR vs {self.label}"


DEFAULT = Reference()


def _first_exec(result) -> pd.Timestamp:
    return result.decisions["execution_date"].iloc[0] if not result.decisions.empty else result.nav.index[0]


def _scores(nav: pd.Series, base: pd.Series, ref: Reference = DEFAULT) -> dict:
    r, b = nav.pct_change().dropna(), base.pct_change().dropna()
    ppy = periods_per_year(nav.index)
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    cagr_b = (base.iloc[-1] / base.iloc[0]) ** (1 / years) - 1
    return {
        "CAGR": cagr,
        "volatilité": float(r.std(ddof=1) * math.sqrt(ppy)),
        "Sharpe": float(r.mean() / r.std(ddof=1) * math.sqrt(ppy)) if r.std() > 0 else float("nan"),
        "pire chute": max_drawdown(nav)[0],
        ref.cagr_col: cagr - cagr_b,
        ref.ce_col: certainty_equivalent(r, ref.gamma, ppy) - certainty_equivalent(b, ref.gamma, ppy),
    }


def _pair(prices, rule: Callable, config: BacktestConfig, baseline_cache: dict | None = None,
          ref: Reference = DEFAULT):
    strat = run_backtest(prices, rule, config)
    key = (config.rebalance if isinstance(config.rebalance, str) else "custom",
           config.lookback, config.fee_rate, config.cash_rate, config.execution_delay)
    if baseline_cache is not None and key in baseline_cache:
        base = baseline_cache[key]
    else:
        base = run_backtest(prices, ref.rule(), config)
        if baseline_cache is not None:
            baseline_cache[key] = base
    start = max(_first_exec(strat), _first_exec(base))
    return strat.nav.loc[start:], base.nav.loc[start:], strat


# ---------------------------------------------------------------------- parameters
def parameter_sensitivity(prices, factory: Callable, base_params: dict, grid: dict[str, list],
                          config: BacktestConfig, ref: Reference = DEFAULT) -> pd.DataFrame:
    """Every combination of the grid; the base configuration is marked.

    A robust strategy shows a PLATEAU (neighbours behave alike). A lucky one shows a
    PEAK: one combination shines, its neighbours do not.
    """
    cache: dict = {}
    names = list(grid)
    rows = []
    for combo in itertools.product(*(grid[n] for n in names)):
        params = apply_settings(base_params, dict(zip(names, combo)))
        nav, base, _ = _pair(prices, factory(**params), config, cache, ref)
        row = {n: v for n, v in zip(names, combo)}
        base_values = current_settings(base_params, names)
        row["configuration de base"] = all(base_values.get(n) == v for n, v in zip(names, combo))
        rows.append({**row, **_scores(nav, base, ref)})
    return pd.DataFrame(rows)


def plateau_summary(table: pd.DataFrame, metric: str | None = None) -> dict:
    metric = metric or next(c for c in table.columns if c.startswith("écart CE"))
    label = metric.split(" vs ", 1)[1]
    values = table[metric]
    base = table.loc[table["configuration de base"], metric]
    return {
        "combinaisons testées": int(len(table)),
        f"% des combinaisons qui battent {label}": float((values > 0).mean()),
        "médiane de l'écart": float(values.median()),
        "pire combinaison": float(values.min()),
        "meilleure combinaison": float(values.max()),
        "rang de la configuration de base (1 = meilleure)":
            int((values > float(base.iloc[0])).sum() + 1) if len(base) else None,
    }


# ---------------------------------------------------------------------------- costs
def cost_sensitivity(prices, rule_factory: Callable[[], Callable], config: BacktestConfig,
                     fee_rates=(0.0, 0.0005, 0.001, 0.0025, 0.005, 0.01),
                     ref: Reference = DEFAULT) -> tuple[pd.DataFrame, float]:
    """The same strategy under rising costs, and the cost at which its edge vanishes."""
    rows = []
    for fee in fee_rates:
        cfg = replace(config, fee_rate=fee)
        nav, base, strat = _pair(prices, rule_factory(), cfg, ref=ref)
        rows.append({"frais par transaction": fee, **_scores(nav, base, ref),
                     "frais payés": strat.fees_paid})
    table = pd.DataFrame(rows)
    breakeven = breakeven_cost(table["frais par transaction"].to_numpy(),
                               table[ref.ce_col].to_numpy())
    return table, breakeven


def breakeven_cost(fees, gaps) -> float:
    """Cost at which the edge vanishes, by linear interpolation between grid points.
    0 if there is no edge even without costs; inf if the edge survives every cost tested."""
    fees, gaps = np.asarray(fees, float), np.asarray(gaps, float)
    if gaps[0] <= 0:
        return 0.0
    for i in range(1, len(gaps)):
        if gaps[i] <= 0 < gaps[i - 1]:
            return float(fees[i - 1] + (fees[i] - fees[i - 1]) * gaps[i - 1] / (gaps[i - 1] - gaps[i]))
    return float("inf")


# ----------------------------------------------------------------------- rebalancing
def rebalance_sensitivity(prices, rule_factory: Callable[[], Callable], config: BacktestConfig,
                          frequencies=("weekly", "monthly", "quarterly", "yearly"),
                          ref: Reference = DEFAULT) -> pd.DataFrame:
    rows = []
    for freq in frequencies:
        cfg = replace(config, rebalance=freq)
        nav, base, strat = _pair(prices, rule_factory(), cfg, ref=ref)
        rows.append({"rebalancement": freq, **_scores(nav, base, ref), "décisions": strat.n_decisions,
                     "frais payés": strat.fees_paid})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- entry / rolling
def entry_date_sensitivity(nav: pd.Series, base: pd.Series, ref: Reference = DEFAULT) -> pd.DataFrame:
    """An investor entering on the first trading day of each year, holding to the end."""
    rows = []
    for year in sorted(set(nav.index.year))[:-1]:
        start = nav.index[nav.index.year == year][0]
        if (nav.index[-1] - start).days < 365:
            continue
        rows.append({"entrée": start.date(), **_scores(nav.loc[start:], base.loc[start:], ref)})
    return pd.DataFrame(rows)


def rolling_windows(nav: pd.Series, base: pd.Series, years: int = 3, step: str = "ME",
                    ref: Reference = DEFAULT) -> pd.DataFrame:
    """Every window of ``years`` years, moved month by month."""
    rows = []
    for start in nav.resample(step).first().index:
        start = nav.index[nav.index.searchsorted(start)] if start < nav.index[-1] else None
        if start is None:
            continue
        end = start + pd.DateOffset(years=years)
        if end > nav.index[-1]:
            break
        s, b = nav.loc[start:end], base.loc[start:end]
        rows.append({"début": start.date(), "fin": s.index[-1].date(), **_scores(s, b, ref)})
    return pd.DataFrame(rows)


def rolling_summary(table: pd.DataFrame) -> dict:
    if table.empty:
        return {"fenêtres": 0}
    gap_col = next(c for c in table.columns if c.startswith("écart CAGR"))
    ce_col = next(c for c in table.columns if c.startswith("écart CE"))
    label, gamma = gap_col.split(" vs ", 1)[1], ce_col.split("γ=")[1].split(" ")[0]
    gap, ce = table[gap_col], table[ce_col]
    return {
        "fenêtres": int(len(table)),
        f"% des fenêtres où la stratégie bat {label}": float((gap > 0).mean()),
        "écart médian": float(gap.median()),
        "pire fenêtre": float(gap.min()),
        "meilleure fenêtre": float(gap.max()),
        # the same question once risk is priced in, as in the lab's verdict: a defensive
        # strategy can lose on raw return and still be the better deal
        f"% des fenêtres où la stratégie bat {label} ajusté du risque (CE γ={gamma})": float((ce > 0).mean()),
        "écart CE médian": float(ce.median()),
    }


# ------------------------------------------------------------------------ bootstrap
def block_bootstrap(nav: pd.Series, base: pd.Series, *, n_paths: int = 2000, block_months: int = 6,
                    seed: int = 0, ref: Reference = DEFAULT) -> pd.DataFrame:
    """Reshuffles history by blocks of months, keeping strategy and 1/N PAIRED.

    Blocks preserve short-term dependence (a crisis stays a crisis); pairing keeps both
    portfolios in the same market, so the gap reflects the strategy, not the draw.
    """
    def monthly(x):
        return x.resample("ME").last().pct_change().dropna().to_numpy()

    rs, rb = monthly(nav), monthly(base)
    n = min(rs.size, rb.size)
    rs, rb = rs[-n:], rb[-n:]
    if n < 2 * block_months:
        raise ValueError("not enough months for a block bootstrap")
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block_months)
    starts = rng.integers(0, n, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block_months)[None, None, :]) % n   # circular
    idx = idx.reshape(n_paths, -1)[:, :n]
    ps, pb = rs[idx], rb[idx]
    years = n / 12
    cagr_s = np.prod(1 + ps, axis=1) ** (1 / years) - 1
    cagr_b = np.prod(1 + pb, axis=1) ** (1 / years) - 1
    sharpe_s = ps.mean(axis=1) / ps.std(axis=1, ddof=1) * math.sqrt(12)
    mdd_s = np.array([max_drawdown(pd.Series(np.cumprod(1 + p)))[0] for p in ps])

    def ce(p, gamma=ref.gamma):  # annualised CRRA certainty equivalent of each path
        if abs(gamma - 1) < 1e-12:
            return np.exp(np.mean(np.log1p(p), axis=1)) ** 12 - 1
        return (np.mean((1 + p) ** (1 - gamma), axis=1) ** (1 / (1 - gamma))) ** 12 - 1
    rows = []
    for name, values in (("CAGR stratégie", cagr_s), (ref.cagr_col, cagr_s - cagr_b),
                         (ref.ce_col, ce(ps) - ce(pb)),
                         ("Sharpe stratégie (mensuel annualisé)", sharpe_s),
                         ("pire chute stratégie", mdd_s)):
        q = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
        rows.append({"mesure": name, "5 %": q[0], "25 %": q[1], "médiane": q[2], "75 %": q[3],
                     "95 %": q[4], "probabilité > 0": float((values > 0).mean())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ all
def full_robustness(prices, factory: Callable, params: dict, config: BacktestConfig, *,
                    grid: dict[str, list] | None = None, rolling_years: int = 3,
                    ref: Reference = DEFAULT) -> dict:
    """Every robustness table for one strategy."""
    def rule():
        return factory(**params)

    nav, base, _ = _pair(prices, rule(), config, ref=ref)
    out = {}
    if grid:
        table = parameter_sensitivity(prices, factory, params, grid, config, ref)
        out["paramètres"] = table
        out["paramètres - synthèse"] = pd.Series(plateau_summary(table), name="valeur").to_frame()
    costs, breakeven = cost_sensitivity(prices, rule, config, ref=ref)
    out["frais"] = costs
    out["frais - seuil de rentabilité"] = breakeven
    out["rebalancement"] = rebalance_sensitivity(prices, rule, config, ref=ref)
    out["date d'entrée"] = entry_date_sensitivity(nav, base, ref)
    windows = rolling_windows(nav, base, rolling_years, ref=ref)
    out[f"fenêtres {rolling_years} ans"] = windows
    out[f"fenêtres {rolling_years} ans - synthèse"] = pd.Series(rolling_summary(windows), name="valeur").to_frame()
    out["bootstrap"] = block_bootstrap(nav, base, ref=ref)
    return out
