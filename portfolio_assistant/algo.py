"""Algorithmic mode: signals, market and limit orders, stops and targets, one line at a time.

The lab's strategies hold a portfolio and rebalance it; here each line has its own entries
and exits, like a trading system. Same honesty rules as the rest of the package:

* a signal computed at a close only uses prices up to that close (rolling windows look
  back, never forward), and its order is executed at the NEXT close at the earliest;
* closing prices only: a limit order fills at the close of a session that ends beyond
  its limit (at that close); a move through the limit inside the session is not seen,
  so fills are rarer than in reality, never luckier;
* a stop or a target is checked at the close and executed at the next close, the slippage
  of a stop in a falling market included;
* costs, leverage and shorts follow ``execution`` (same cost models, same financing);
* the accounting identity is checked on every bar.

Not modelled: intraday highs and lows, gaps inside the session, partial fills.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .data import validate_prices
from .engine import IDENTITY_TOLERANCE, BacktestError
from .execution import CostModel, Instrument, guess_instrument

SIGNALS = {
    "croisement": "Croisement de moyennes mobiles (courte au-dessus de la longue : achat)",
    "cassure": "Cassure : clôture au-dessus du plus haut des N dernières séances",
    "rsi": "RSI : achat en survente, vente en surachat (retour à la moyenne)",
    "zscore": "Écart à la moyenne : achat quand le cours est anormalement bas",
}


@dataclass
class AlgoSpec:
    name: str = "Système"
    signal: str = "croisement"
    fast: int = 50                 # croisement: short average; cassure: N; rsi/zscore: period
    slow: int = 200                # croisement: long average; cassure: exit window
    low: float = 30.0              # rsi: oversold; zscore: entry threshold (in std)
    high: float = 70.0             # rsi: overbought
    short: bool = False            # also take short signals
    order: str = "marché"          # marché | limite
    limit_offset: float = 0.01     # buy 1 % under the signal close (sell 1 % above)
    limit_days: int = 5            # an unfilled limit order is cancelled after this many sessions
    stop_loss: float | None = 0.10
    take_profit: float | None = None
    trailing: float | None = None
    max_days: int | None = None
    size: float = 0.20             # each position: this fraction of the equity
    max_positions: int = 5
    max_gross: float = 1.0
    initial_capital: float = 100_000.0
    cash_rate: float = 0.0
    debit_spread: float = 0.015
    borrow_fee: float = 0.005
    costs: CostModel = field(default_factory=CostModel)

    def __post_init__(self):
        if self.signal not in SIGNALS:
            raise ValueError(f"signal inconnu : {self.signal!r}")
        if self.order not in ("marché", "limite"):
            raise ValueError("ordre « marché » ou « limite »")
        if not 0 < self.size <= self.max_gross:
            raise ValueError("la taille d'une position doit être entre 0 et l'exposition autorisée")
        for name in ("stop_loss", "take_profit", "trailing"):
            v = getattr(self, name)
            if v is not None and not 0 < v < 1:
                raise ValueError(f"{name} doit être entre 0 et 100 %")
        if self.signal == "croisement" and self.fast >= self.slow:
            raise ValueError("la moyenne courte doit être plus courte que la longue")

    @property
    def warmup(self) -> int:
        return int(max(self.fast, self.slow if self.signal in ("croisement", "cassure") else 0)) + 1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["costs"] = asdict(self.costs)
        return d


# --------------------------------------------------------------------------- signals
def _rsi(p: pd.DataFrame, n: int) -> pd.DataFrame:
    d = p.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def signals(prices: pd.DataFrame, spec: AlgoSpec) -> dict[str, pd.DataFrame]:
    """Boolean frames: entry_long, entry_short, exit_long, exit_short. Each value at a date
    only depends on prices up to that date."""
    p = prices
    if spec.signal == "croisement":
        diff = p.rolling(spec.fast).mean() - p.rolling(spec.slow).mean()
        prev = diff.shift(1)
        up, down = (diff > 0) & (prev <= 0), (diff < 0) & (prev >= 0)
        out = {"entry_long": up, "entry_short": down, "exit_long": down, "exit_short": up}
    elif spec.signal == "cassure":
        hi, lo = p.rolling(spec.fast).max().shift(1), p.rolling(spec.fast).min().shift(1)
        hi_x, lo_x = p.rolling(spec.slow).max().shift(1), p.rolling(spec.slow).min().shift(1)
        out = {"entry_long": p > hi, "entry_short": p < lo, "exit_long": p < lo_x, "exit_short": p > hi_x}
    elif spec.signal == "rsi":
        r = _rsi(p, spec.fast)
        prev = r.shift(1)
        out = {"entry_long": (r < spec.low) & (prev >= spec.low), "entry_short": (r > spec.high) & (prev <= spec.high),
               "exit_long": r > 50, "exit_short": r < 50}
    else:
        z = (p - p.rolling(spec.fast).mean()) / p.rolling(spec.fast).std()
        prev = z.shift(1)
        out = {"entry_long": (z < -spec.low) & (prev >= -spec.low), "entry_short": (z > spec.low) & (prev <= spec.low),
               "exit_long": z > 0, "exit_short": z < 0}
    return {k: v.fillna(False).astype(bool) for k, v in out.items()}


# ---------------------------------------------------------------------------- engine
@dataclass
class AlgoResult:
    nav: pd.Series
    cash: pd.Series
    positions: pd.DataFrame          # units held at each close
    weights: pd.DataFrame
    journal: pd.DataFrame            # one row per position, entry to exit
    orders: pd.DataFrame             # every order: placed, filled, cancelled, refused
    costs: dict
    spec: AlgoSpec
    prices: pd.DataFrame

    def stats(self) -> dict:
        j = self.journal[self.journal["statut"] == "close"]
        wins, losses = j[j["résultat €"] > 0], j[j["résultat €"] <= 0]
        gross_win, gross_loss = float(wins["résultat €"].sum()), float(-losses["résultat €"].sum())
        return {"positions fermées": int(len(j)),
                "positions ouvertes": int((self.journal["statut"] == "ouverte").sum()),
                "gagnantes": float(len(wins) / len(j)) if len(j) else float("nan"),
                "gain moyen": float(wins["résultat %"].mean()) if len(wins) else float("nan"),
                "perte moyenne": float(losses["résultat %"].mean()) if len(losses) else float("nan"),
                "facteur de profit": gross_win / gross_loss if gross_loss > 0 else float("inf"),
                "espérance par position": float(j["résultat €"].mean()) if len(j) else float("nan"),
                "durée moyenne (séances)": float(j["séances"].mean()) if len(j) else float("nan"),
                "ordres à cours limité non exécutés": int((self.orders["statut"] == "annulé").sum())
                if not self.orders.empty else 0}


def run_algo(prices: pd.DataFrame, spec: AlgoSpec, instruments: dict | None = None,
             ranges: dict[str, pd.DataFrame] | None = None) -> AlgoResult:
    """``ranges`` ({"open", "high", "low"} frames in the same currency as ``prices``) turns on
    execution inside the session: a limit order fills at its limit when the day's range
    reaches it (at the open when the market opens beyond it), and a stop-loss, target or
    trailing stop is executed on the day it is touched, at its level or at the open after a
    gap. When the stop and the target are both inside one day's range, the stop is assumed
    first. A day without a range falls back to the close."""
    prices = validate_prices(prices)
    if prices.isna().any().any():
        raise ValueError("the backtest needs a full panel; use data.common_window() first")
    index, tickers = prices.index, list(prices.columns)
    px = prices.to_numpy()
    n_bars, n_assets = px.shape
    if n_bars <= spec.warmup + 2:
        raise ValueError("pas assez d'historique pour ce signal")
    given = dict(instruments or {})
    instr = [given[t] if isinstance(given.get(t), Instrument) else guess_instrument(t, given.get(t)) for t in tickers]
    sig = {k: v.to_numpy() for k, v in signals(prices, spec).items()}
    sigma = prices.pct_change().rolling(20, min_periods=5).std().shift(1).to_numpy()
    day_gap = np.r_[0.0, np.diff(index.to_numpy(dtype="datetime64[ns]")).astype("int64") / 86_400e9]
    borrows = spec.max_gross > 1.0 or spec.short
    if ranges:
        O, HI, LO = (ranges[k].reindex(index)[tickers].to_numpy(dtype=float) for k in ("open", "high", "low"))
        intraday = np.isfinite(O) & np.isfinite(HI) & np.isfinite(LO)
    else:
        intraday = np.zeros((n_bars, n_assets), dtype=bool)
    exec_bar = np.zeros(n_bars)

    cash_now = float(spec.initial_capital)
    units = np.zeros(n_assets)
    hist_units = np.zeros((n_bars, n_assets))
    cash = np.empty(n_bars)
    carry_bar, cost_bar = np.zeros(n_bars), np.zeros(n_bars)
    totals = {k: 0.0 for k in ("commission", "spread", "slippage", "impact", "tax", "fx", "debit", "borrow", "credit")}
    open_pos: dict[int, dict] = {}         # j -> position
    pending: list[dict] = []               # orders waiting for a fill
    journal, orders = [], []

    def fill(t: int, j: int, notional: float, at: float | None = None) -> float:
        nonlocal cash_now
        at = px[t, j] if at is None else at
        parts = spec.costs.line_costs(np.array([notional]), [instr[j]], sigma[t, j:j + 1], index[t])
        cost = float(sum(p[0] for p in parts.values()))
        for k, v in parts.items():
            totals[k] += float(v[0])
        du = notional / at
        units[j] += du
        cash_now -= notional + cost
        cost_bar[t] += cost
        exec_bar[t] += du * (px[t, j] - at)          # filled away from the close: P&L of the session
        return cost

    def close_position(t: int, j: int, at: float, reason: str) -> float:
        pos = open_pos.pop(j)
        cost = fill(t, j, -units[j] * at, at)
        pnl = pos["sens"] * (at - pos["prix"]) * abs(pos["unités"]) - pos["frais"] - cost
        journal.append({"ticker": tickers[j], "sens": "achat" if pos["sens"] > 0 else "vente à découvert",
                        "entrée": index[pos["t"]], "prix d'entrée": pos["prix"], "sortie": index[t],
                        "prix de sortie": at, "motif": reason, "séances": t - pos["t"],
                        "résultat €": pnl, "résultat %": pos["sens"] * (at / pos["prix"] - 1), "statut": "close"})
        return cost

    for t in range(n_bars):
        # 1. carry over the bar
        if t > 0:
            years = day_gap[t] / 365.0
            held = units * px[t - 1]
            short_value = float(-held[held < 0].sum())
            growth = (1 + spec.cash_rate) ** years - 1 if spec.cash_rate else 0.0
            credit = max(cash_now - short_value, 0.0) * growth
            debit = max(-cash_now, 0.0) * (growth + (1 + spec.debit_spread) ** years - 1)
            borrow = short_value * ((1 + spec.borrow_fee) ** years - 1)
            carry = credit - debit - borrow
            cash_now += carry
            carry_bar[t] = carry
            totals["credit"] += credit
            totals["debit"] += debit
            totals["borrow"] += borrow

        # 2. orders placed at an earlier close
        still = []
        for o in pending:
            j, price = o["j"], px[t, o["j"]]
            if o["type"] == "limite" and intraday[t, j]:
                lim = o["limite"]
                if o["sens"] > 0:
                    crossed, price = LO[t, j] <= lim, min(O[t, j], lim)
                else:
                    crossed, price = HI[t, j] >= lim, max(O[t, j], lim)
            elif o["type"] == "limite":
                crossed = price <= o["limite"] if o["sens"] > 0 else price >= o["limite"]
            if o["type"] == "limite":
                if not crossed:
                    if t - o["placed"] >= spec.limit_days:
                        o.update(statut="annulé", date_statut=index[t])
                        orders.append(o)
                    else:
                        still.append(o)
                    continue
            if o["but"] == "sortie":
                if j not in open_pos:
                    continue
                close_position(t, j, price, o["motif"])
            else:
                equity = cash_now + float((units * px[t]).sum())
                gross = float(np.abs(units * px[t]).sum())
                value = min(spec.size * equity, spec.max_gross * equity - gross)
                if not borrows and value > 0:
                    # the purchase AND its costs must fit in the cash
                    for _ in range(20):
                        parts = spec.costs.line_costs(np.array([o["sens"] * value]), [instr[j]],
                                                      sigma[t, j:j + 1], index[t])
                        cost = float(sum(p[0] for p in parts.values()))
                        if value + cost <= cash_now + 1e-9:
                            break
                        value = max(cash_now - cost, 0.0) * (1 - 1e-12)
                if value <= 1e-6 * spec.initial_capital or j in open_pos:
                    o.update(statut="refusé", date_statut=index[t], motif="exposition ou liquidités insuffisantes")
                    orders.append(o)
                    continue
                cost = fill(t, j, o["sens"] * value, price)
                open_pos[j] = {"sens": o["sens"], "t": t, "prix": price, "unités": value / price, "frais": cost,
                               "extrême": price}
            o.update(statut="exécuté", date_statut=index[t], prix_exécution=price)
            orders.append(o)
        pending = still

        # 2b. stops and targets inside the session, on positions opened before today
        exiting = {o["j"] for o in pending if o["but"] == "sortie"}
        for j in [j for j, pos in open_pos.items() if pos["t"] < t and intraday[t, j] and j not in exiting]:
            pos, side = open_pos[j], open_pos[j]["sens"]
            worst, best = (LO[t, j], HI[t, j]) if side > 0 else (HI[t, j], LO[t, j])
            stops = []
            if spec.stop_loss:
                stops.append((pos["prix"] * (1 - side * spec.stop_loss), "stop-loss (en séance)"))
            if spec.trailing:
                stops.append((pos["extrême"] * (1 - side * spec.trailing), "stop suiveur (en séance)"))
            hit = [(lvl, why) for lvl, why in stops if (worst <= lvl if side > 0 else worst >= lvl)]
            if hit:
                lvl, why = max(hit) if side > 0 else min(hit)          # the first level met on the way down
                close_position(t, j, min(O[t, j], lvl) if side > 0 else max(O[t, j], lvl), why)
                continue
            if spec.take_profit:
                lvl = pos["prix"] * (1 + side * spec.take_profit)
                if (best >= lvl if side > 0 else best <= lvl):
                    close_position(t, j, max(O[t, j], lvl) if side > 0 else min(O[t, j], lvl), "objectif atteint (en séance)")
                    continue
            pos["extrême"] = max(pos["extrême"], best) if side > 0 else min(pos["extrême"], best)

        # 3. book the bar
        hist_units[t] = units
        cash[t] = cash_now
        equity = cash_now + float((units * px[t]).sum())
        if borrows and equity <= 0:
            raise BacktestError(f"ruine le {index[t].date()} : les pertes dépassent la mise")
        if not borrows and cash_now < -1e-6 * spec.initial_capital:
            raise BacktestError(f"cash went negative ({cash_now:.4f}) on {index[t].date()}")
        if t == n_bars - 1 or t < spec.warmup:
            continue

        # 4. decide at this close: exits first, then entries
        waiting = {(o["j"], o["but"]) for o in pending}
        for j, pos in open_pos.items():
            if (j, "sortie") in waiting:
                continue
            price = px[t, j]
            move = pos["sens"] * (price / pos["prix"] - 1)
            reason = None
            if not intraday[t, j]:                    # with the session's range, done in 2b
                pos["extrême"] = max(pos["extrême"], price) if pos["sens"] > 0 else min(pos["extrême"], price)
            if intraday[t, j]:
                pass
            elif spec.stop_loss and move <= -spec.stop_loss:
                reason = "stop-loss"
            elif spec.take_profit and move >= spec.take_profit:
                reason = "objectif atteint"
            elif spec.trailing and pos["sens"] * (price / pos["extrême"] - 1) <= -spec.trailing:
                reason = "stop suiveur"
            if reason:
                pass
            elif spec.max_days and t - pos["t"] >= spec.max_days:
                reason = "durée maximale"
            elif (sig["exit_long"][t, j] if pos["sens"] > 0 else sig["exit_short"][t, j]):
                reason = "signal de sortie"
            if reason:
                pending.append({"j": j, "ticker": tickers[j], "but": "sortie", "sens": -pos["sens"], "type": "marché",
                                "placé le": index[t], "placed": t, "motif": reason})
        slots = spec.max_positions - len(open_pos) - sum(o["but"] == "entrée" for o in pending)
        for j in range(n_assets):
            if slots <= 0:
                break
            if j in open_pos or (j, "entrée") in waiting:
                continue
            side = 1 if sig["entry_long"][t, j] else (-1 if spec.short and sig["entry_short"][t, j] else 0)
            if not side:
                continue
            o = {"j": j, "ticker": tickers[j], "but": "entrée", "sens": side, "type": spec.order, "placé le": index[t],
                 "placed": t, "motif": SIGNALS[spec.signal].split(" ")[0].split(":")[0]}
            if spec.order == "limite":
                o["limite"] = px[t, j] * (1 - side * spec.limit_offset)
            pending.append(o)
            slots -= 1

    for o in pending:
        o.update(statut="en attente", date_statut=index[-1])
        orders.append(o)
    # positions still open: valued at the last close
    for j, pos in open_pos.items():
        price = px[-1, j]
        journal.append({"ticker": tickers[j], "sens": "achat" if pos["sens"] > 0 else "vente à découvert",
                        "entrée": index[pos["t"]], "prix d'entrée": pos["prix"], "sortie": pd.NaT,
                        "prix de sortie": price, "motif": "", "séances": n_bars - 1 - pos["t"],
                        "résultat €": pos["sens"] * (price - pos["prix"]) * abs(pos["unités"]) - pos["frais"],
                        "résultat %": pos["sens"] * (price / pos["prix"] - 1), "statut": "ouverte"})

    nav = cash + (hist_units * px).sum(axis=1)
    explained = (hist_units[:-1] * (px[1:] - px[:-1])).sum(axis=1) + carry_bar[1:] - cost_bar[1:] + exec_bar[1:]
    gap = np.abs(np.diff(nav) - explained)
    if gap.size and gap.max() > IDENTITY_TOLERANCE * max(spec.initial_capital, float(np.abs(nav).max())):
        raise BacktestError(f"P&L identity broken on {index[int(np.argmax(gap)) + 1].date()}")
    nav_s = pd.Series(nav, index=index, name="nav")
    cols = ["ticker", "sens", "entrée", "prix d'entrée", "sortie", "prix de sortie", "motif", "séances",
            "résultat €", "résultat %", "statut"]
    order_cols = ["ticker", "but", "sens", "type", "placé le", "limite", "statut", "date_statut", "prix_exécution", "motif"]
    ord_df = pd.DataFrame(orders).reindex(columns=order_cols) if orders else pd.DataFrame(columns=order_cols)
    return AlgoResult(nav=nav_s, cash=pd.Series(cash, index=index, name="cash"),
                      positions=pd.DataFrame(hist_units, index=index, columns=tickers),
                      weights=pd.DataFrame(hist_units * px, index=index, columns=tickers).div(nav_s, axis=0),
                      journal=pd.DataFrame(journal, columns=cols).sort_values("entrée", ignore_index=True),
                      orders=ord_df, costs=totals, spec=spec, prices=prices)
