"""Research mode: what a signal is worth before any backtest, and blind practice.

``study`` measures every occurrence of an entry signal: the return over several horizons
from the NEXT close (the first price one could trade at), the best and worst excursion
during the holding window (MFE / MAE), and, with a target and a stop, which one was
touched first. Each figure is compared with the same horizon taken on every day, so a
signal only "works" if it beats simply being in the market. Occurrences too close to the
end of the data are counted as censored, never dropped silently.

``Blind`` is a practice session: a random window of the construction period, the asset's
name and the dates hidden, one decision at a time; the future is never in the object.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .algo import AlgoSpec, signals


def study(prices: pd.DataFrame, spec: AlgoSpec, *, horizons=(5, 21, 63), target: float | None = None,
          stop: float | None = None, window: int = 63, n_boot: int = 1000, seed: int = 0) -> dict:
    if prices.isna().any().any():
        raise ValueError("cours incomplets : utiliser un panneau sans trous")
    sig = signals(prices, spec)
    px = prices.to_numpy()
    n = len(prices)
    rows = []
    sides = [("achat", 1, sig["entry_long"].to_numpy())]
    if spec.short:
        sides.append(("vente à découvert", -1, sig["entry_short"].to_numpy()))
    for label, side, mask in sides:
        for t, j in zip(*np.nonzero(mask)):
            if t < spec.warmup:
                continue
            e = t + 1                                         # executed at the next close
            row = {"date": prices.index[t], "ticker": prices.columns[j], "sens": label}
            if e >= n:
                row["censuré"] = True
                rows.append(row)
                continue
            p0 = px[e, j]
            for h in horizons:
                row[f"{h} j"] = side * (px[e + h, j] / p0 - 1) if e + h < n else np.nan
            path = side * (px[e: min(e + window, n - 1) + 1, j] / p0 - 1)
            row["MFE"], row["MAE"] = float(path.max()), float(path.min())
            if target or stop:
                hit_t = np.flatnonzero(path >= target)[0] if target and (path >= target).any() else None
                hit_s = np.flatnonzero(path <= -stop)[0] if stop and (path <= -stop).any() else None
                if hit_t is None and hit_s is None:
                    row["premier touché"] = "aucun (censuré)" if e + window >= n else "aucun"
                elif hit_s is None or (hit_t is not None and hit_t < hit_s):
                    row["premier touché"], row["séances"] = "objectif", int(hit_t)
                else:
                    row["premier touché"], row["séances"] = "stop", int(hit_s)
            row["censuré"] = e + max(horizons) >= n
            rows.append(row)
    events = pd.DataFrame(rows)
    if events.empty:
        return {"events": events, "table": pd.DataFrame(), "n": 0}
    # every day, every line: what simply being in the market gave over the same horizons
    rng = np.random.default_rng(seed)
    table = []
    months = events["date"].dt.to_period("M").astype(str).to_numpy()
    for h in horizons:
        col = f"{h} j"
        base = (prices.shift(-h - 1) / prices.shift(-1) - 1).iloc[spec.warmup:].to_numpy().ravel()
        base = base[np.isfinite(base)]
        x = events[col].to_numpy(dtype=float)
        ok = np.isfinite(x)
        side = np.where(events["sens"] == "achat", 1.0, -1.0)
        ref = float(np.mean(base))
        edge_each = x[ok] - side[ok] * ref                    # a short is judged against minus the market
        # bootstrap by calendar month: occurrences of the same month move together
        groups = pd.Series(edge_each).groupby(months[ok]).apply(list).tolist()
        boots = []
        for _ in range(n_boot):
            pick = rng.integers(0, len(groups), len(groups))
            boots.append(np.mean(np.concatenate([groups[i] for i in pick])))
        lo, hi = np.percentile(boots, [5, 95]) if boots else (np.nan, np.nan)
        table.append({"horizon": col, "occurrences": int(ok.sum()), "moyenne": float(np.mean(x[ok])) if ok.any() else np.nan,
                      "médiane": float(np.median(x[ok])) if ok.any() else np.nan,
                      "gagnantes": float(np.mean(x[ok] > 0)) if ok.any() else np.nan,
                      "marché, tous les jours": ref, "avance": float(np.mean(edge_each)) if ok.any() else np.nan,
                      "avance, 90 % bas": float(lo), "avance, 90 % haut": float(hi),
                      "verdict": ("avance" if lo > 0 else "retard" if hi < 0 else "indiscernable du marché")})
    out = {"events": events, "table": pd.DataFrame(table), "n": int(len(events)),
           "censored": int(events["censuré"].sum())}
    if "premier touché" in events:
        out["touches"] = events["premier touché"].value_counts().to_dict()
    return out


# ------------------------------------------------------------------------- blind practice
@dataclass
class Blind:
    """A practice window. ``visible()`` never returns a price after the cursor."""
    series: pd.Series                 # the whole window, kept private to the session
    shown: int = 120                  # history visible at the start
    steps: int = 60                   # decisions to take
    cursor: int = 0
    position: int = 0
    nav: list = field(default_factory=lambda: [100.0])
    log: list = field(default_factory=list)
    fee: float = 0.001
    step_bars: int = 5

    @classmethod
    def draw(cls, prices: pd.DataFrame, seed: int, shown: int = 120, steps: int = 60, step_bars: int = 5) -> "Blind":
        rng = np.random.default_rng(seed)
        need = shown + steps * step_bars + 1
        if len(prices) <= need:
            raise ValueError("période trop courte pour un entraînement")
        j = int(rng.integers(0, prices.shape[1]))
        start = int(rng.integers(0, len(prices) - need))
        s = prices.iloc[start: start + need, j].reset_index(drop=True)
        b = cls(series=s / s.iloc[0] * 100, shown=shown, steps=steps)
        b.cursor, b.step_bars = shown - 1, step_bars
        b.ticker, b.dates = prices.columns[j], (prices.index[start], prices.index[start + need - 1])
        return b

    def visible(self) -> pd.Series:
        return self.series.iloc[: self.cursor + 1]

    @property
    def done(self) -> bool:
        return len(self.log) >= self.steps

    def decide(self, position: int) -> None:
        """New position (1 bought, 0 out, −1 short) from the next close, held for ``step_bars``."""
        if self.done:
            return
        cost = self.fee * abs(position - self.position)
        a, b = self.cursor, self.cursor + self.step_bars
        move = float(self.series.iloc[b] / self.series.iloc[a] - 1)
        self.nav.append(self.nav[-1] * (1 - cost) * (1 + position * move))
        self.log.append({"pas": len(self.log) + 1, "position": position, "variation de l'actif": move})
        self.position, self.cursor = position, b

    def score(self) -> dict:
        start = self.shown - 1
        hold = float(self.series.iloc[self.cursor] / self.series.iloc[start] - 1)
        mine = self.nav[-1] / 100 - 1
        right = [r for r in self.log if r["position"] != 0]
        good = sum((r["position"] > 0) == (r["variation de l'actif"] > 0) for r in right)
        return {"vous": mine, "rester investi": hold, "écart": mine - hold,
                "décisions dans le bon sens": good / len(right) if right else float("nan"), "pas": len(self.log)}
