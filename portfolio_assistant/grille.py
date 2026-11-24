"""A grid of settings for the algorithmic mode, run on every core of the machine.

What makes it honest rather than a search for the luckiest setting:

* every combination is a configuration tried, and the workshop counts each one;
* the best one is chosen on the first 70 % of the construction period only, then replayed
  on the last 30 %, which it has not seen: the page shows how much of it is left;
* its Sharpe is deflated by the number of combinations (Bailey & López de Prado): the
  Sharpe the best of N would reach by pure chance is shown next to it.
"""

from __future__ import annotations

import itertools
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd

from . import algo
from .analytics import sharpe_statistics
from .metrics import summary

MAX_COMBINATIONS = 2000
MAX_EN_LIGNE = 500                 # a shared server: smaller grids, one core by default (PA_GRILLE_COEURS)


def online() -> bool:
    return os.environ.get("PA_EN_LIGNE", "").strip().lower() in ("1", "oui", "true", "yes")


def limit() -> int:
    return MAX_EN_LIGNE if online() else MAX_COMBINATIONS
SPLIT = 0.7
GRID_PARAMS = {
    "fast": "Fenêtre principale (séances)", "slow": "Fenêtre de sortie ou longue (séances)",
    "low": "Seuil d'entrée", "stop_loss": "Stop-loss", "take_profit": "Objectif", "trailing": "Stop suiveur",
    "size": "Taille d'une position", "max_positions": "Positions simultanées",
}

_PRICES: pd.DataFrame | None = None
_INSTRUMENTS: dict | None = None


def _init(prices, instruments):
    global _PRICES, _INSTRUMENTS
    _PRICES, _INSTRUMENTS = prices, instruments


def _one(job):
    """One combination, in a worker: the full period and its two parts."""
    i, spec = job
    out = {"i": i}
    try:
        res = algo.run_algo(_PRICES, spec, _INSTRUMENTS)
        nav = res.nav.iloc[spec.warmup:]
        cut = nav.index[int(len(nav) * SPLIT)]
        for label, part in (("", nav), ("_a", nav.loc[:cut]), ("_b", nav.loc[cut:])):
            s = summary(part)
            out.update({f"cagr{label}": s["cagr"], f"vol{label}": s["volatility"], f"sharpe{label}": s["sharpe"],
                        f"dd{label}": s["max_drawdown"]})
        st = res.stats()
        out.update(trades=st["positions fermées"], win=st["gagnantes"], pf=st["facteur de profit"])
    except (ValueError, algo.BacktestError) as exc:
        out["erreur"] = str(exc)
    return out


def combinations(base: algo.AlgoSpec, grid: dict[str, list]) -> list[tuple[dict, algo.AlgoSpec]]:
    keys = [k for k, v in grid.items() if v]
    unknown = sorted(set(keys) - set(GRID_PARAMS))
    if unknown:
        raise ValueError(f"paramètres de grille inconnus : {unknown}")
    sizes = [len(grid[k]) for k in keys]
    total = int(np.prod(sizes)) if sizes else 1
    if total > limit():
        raise ValueError(f"{total} combinaisons : la grille est limitée à {limit()}"
                         + (" en ligne" if online() else ""))
    out = []
    for values in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, values))
        try:
            out.append((params, replace(base, **params)))
        except ValueError:
            continue                                   # an impossible pair (e.g. short ≥ long) is skipped
    return out


def estimate_seconds(prices: pd.DataFrame, base: algo.AlgoSpec, n: int, workers: int | None = None) -> float:
    """Times one run, scales it: shown before a grid is launched."""
    import time
    start = time.perf_counter()
    algo.run_algo(prices, base)
    one = time.perf_counter() - start
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    w = workers or (1 if online() else min(cores, 8))
    return one * n / max(1, min(w, n)) * 1.1


def run_grid(prices: pd.DataFrame, base: algo.AlgoSpec, grid: dict[str, list], instruments: dict | None = None,
             *, workers: int | None = None, progress=None) -> dict:
    combos = combinations(base, grid)
    if not combos:
        raise ValueError("aucune combinaison possible avec ces réglages")
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    default = 1 if online() else cores
    workers = max(1, min(workers or int(os.environ.get("PA_GRILLE_COEURS", default)), len(combos), 8))
    jobs = [(i, spec) for i, (_, spec) in enumerate(combos)]
    rows = []
    if workers == 1:
        _init(prices, instruments)
        for k, job in enumerate(jobs):
            rows.append(_one(job))
            if progress:
                progress((k + 1) / len(jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(prices, instruments)) as pool:
            for k, row in enumerate(pool.map(_one, jobs, chunksize=max(1, len(jobs) // (workers * 8)))):
                rows.append(row)
                if progress:
                    progress((k + 1) / len(jobs))
    table = pd.DataFrame([{**combos[r["i"]][0], **{k: v for k, v in r.items() if k != "i"}} for r in rows])
    ok = table[table["erreur"].isna()] if "erreur" in table else table
    if ok.empty:
        raise ValueError("aucune combinaison n'a pu tourner")
    best_a = ok["sharpe_a"].idxmax()                   # chosen on the first 70 % only
    best = ok.loc[best_a]
    best_spec = combos[int(best_a)][1]
    res = algo.run_algo(prices, best_spec, instruments)
    first = res.nav.iloc[best_spec.warmup:]
    cut = first.index[int(len(first) * SPLIT)]
    stats = sharpe_statistics(first.loc[:cut], n_trials=len(ok), trial_sharpes=list(ok["sharpe_a"].dropna()))
    return {"table": table, "best": best.to_dict(), "best_params": combos[int(best_a)][0], "n": len(combos),
            "workers": workers, "cut": cut, "dsr": stats.get("Sharpe dégonflé (DSR) - probabilité réelle"),
            "chance": stats.get("Sharpe attendu par chance au meilleur des essais"),
            "rank_b": int((ok["sharpe_b"] > best["sharpe_b"]).sum()) + 1,
            "median_b": float(ok["sharpe_b"].median())}
