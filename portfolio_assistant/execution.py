"""Advanced execution for the lab: realistic trading costs, leverage, short selling,
CFD and forex positions, financing and margin calls.

It lives beside ``engine.run_backtest`` and never replaces it: the strategies frozen and
tracked in the lab keep their own engine, their own costs and their fingerprints. With its
default settings (flat 0.10 % fee, long-only, fully invested at most) ``run_advanced`` gives
exactly the NAV of ``run_backtest`` -- a test checks it.

What is modelled, per traded line:

* commission: a rate with a minimum per order;
* half bid-ask spread, by kind of instrument;
* slippage: a fixed number of basis points;
* market impact: square-root law, ``k * sigma * sqrt(order / daily traded value)``, with
  the volatility of the 20 bars BEFORE the execution (no look-ahead);
* transaction taxes on purchases of shares: French financial transaction tax (TTF, French
  shares with a market cap above 1 bn EUR: 0.2 % from 01/08/2012, 0.3 % from 01/01/2017,
  0.4 % from 01/04/2025) and the UK stamp duty (0.5 %). ETFs, CFDs and forex pay neither.

And per bar, on the positions held over the bar:

* positive cash earns the cash rate (or the money-market fund); short sale proceeds earn
  nothing unless ``short_rebate``;
* negative cash (leverage) pays the cash rate plus ``debit_spread``;
* borrowed shares pay ``borrow_fee`` a year on their value;
* CFDs pay ``cfd_overnight`` a year on their notional, long or short.

Margin: when equity / gross exposure falls below ``maintenance_margin`` the broker sells,
at that bar's close and with costs, back to the leverage allowed; an equity at or below
zero is a ruin: everything is closed and the run stops trading.

Not modelled, and said so: forex carry (interest-rate differential, absent from spot
prices), dividends paid on shorted shares beyond what adjusted prices already contain,
intraday prices (orders fill at the close of the execution bar).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

import numpy as np
import pandas as pd

from .data import validate_prices
from .engine import (IDENTITY_TOLERANCE, BacktestError, BacktestResult, DecisionContext, WeightFn,
                     decision_dates)

# ------------------------------------------------------------------------- instruments
KINDS = {
    # half bid-ask spread (bp) and default daily traded value (EUR) of a liquid line
    "etf": {"libellé": "ETF", "spread_bp": 3.0, "adv": 20e6},
    "action": {"libellé": "Action", "spread_bp": 5.0, "adv": 50e6},
    "obligation": {"libellé": "Obligation / ETF obligataire", "spread_bp": 5.0, "adv": 20e6},
    "forex": {"libellé": "Devise (forex)", "spread_bp": 0.5, "adv": 5e9},
    "cfd": {"libellé": "CFD", "spread_bp": 4.0, "adv": 50e6},
}
TTF_RATES = (("2012-08-01", 0.002), ("2017-01-01", 0.003), ("2025-04-01", 0.004))
UK_STAMP = 0.005


@dataclass(frozen=True)
class Instrument:
    ticker: str
    kind: str = "etf"
    ttf: bool = False          # French share, market cap > 1 bn EUR
    stamp: bool = False        # UK share
    adv: float | None = None   # daily traded value in EUR; None = the kind's default
    currency: str = "EUR"      # quotation currency: buying it from euros pays the conversion fee

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"type d'instrument inconnu : {self.kind!r} ({', '.join(KINDS)})")


def guess_instrument(ticker: str, kind: str | None = None) -> Instrument:
    """A sensible default: ``EURUSD=X`` is forex, anything else an ETF unless told."""
    t = str(ticker).strip().upper()
    if kind is None:
        kind = "forex" if t.endswith("=X") else "etf"
    return Instrument(t, kind)


def ttf_rate(date) -> float:
    rate = 0.0
    for start, r in TTF_RATES:
        if pd.Timestamp(date) >= pd.Timestamp(start):
            rate = r
    return rate


# ------------------------------------------------------------------------------ costs
@dataclass(frozen=True)
class CostModel:
    commission_rate: float = 0.001
    commission_min: float = 0.0        # euros per order line
    spreads: bool = False              # pay the half bid-ask spread of each kind
    slippage_bp: float = 0.0
    impact_k: float = 0.0              # square-root impact coefficient (0.1 is a common value)
    taxes: bool = False                # TTF / stamp duty on share purchases
    fx_fee: float = 0.0                # conversion fee on lines quoted in another currency

    def __post_init__(self):
        for name in ("commission_rate", "commission_min", "slippage_bp", "impact_k", "fx_fee"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} ne peut pas être négatif")
        if self.commission_rate >= 0.5:
            raise ValueError("commission_rate doit être < 50 %")

    def line_costs(self, notional: np.ndarray, instruments: list[Instrument], sigma: np.ndarray,
                   date) -> dict[str, np.ndarray]:
        size = np.abs(notional)
        traded = size > 1e-9
        out = {"commission": np.where(traded, np.maximum(self.commission_rate * size, self.commission_min), 0.0)}
        zero = np.zeros_like(size)
        out["spread"] = (size * np.array([KINDS[i.kind]["spread_bp"] for i in instruments]) / 1e4
                         if self.spreads else zero.copy())
        out["slippage"] = size * self.slippage_bp / 1e4
        if self.impact_k:
            adv = np.array([i.adv or KINDS[i.kind]["adv"] for i in instruments])
            out["impact"] = size * self.impact_k * np.nan_to_num(sigma) * np.sqrt(size / adv)
        else:
            out["impact"] = zero.copy()
        if self.taxes:
            rate = np.array([(ttf_rate(date) if i.ttf else 0.0) + (UK_STAMP if i.stamp else 0.0)
                             if i.kind == "action" else 0.0 for i in instruments])
            out["tax"] = np.where(notional > 0, notional * rate, 0.0)
        else:
            out["tax"] = zero.copy()
        out["fx"] = (size * self.fx_fee * np.array([i.currency.upper() != "EUR" for i in instruments], dtype=float)
                     if self.fx_fee else zero.copy())
        return out


COST_PRESETS: dict[str, CostModel] = {
    "Frais du laboratoire (0,10 % par transaction)": CostModel(0.001),
    "Aucun frais (pour comparer)": CostModel(0.0),
    "Courtier en ligne": CostModel(0.001, 1.0, spreads=True, slippage_bp=2.0, impact_k=0.1, taxes=True, fx_fee=0.0025),
    "Banque de réseau": CostModel(0.005, 10.0, spreads=True, slippage_bp=2.0, impact_k=0.1, taxes=True, fx_fee=0.005),
    "Institutionnel": CostModel(0.0002, 0.0, spreads=True, slippage_bp=1.0, impact_k=0.1, taxes=True, fx_fee=0.0002),
}
COST_NOTES = {
    "Courtier en ligne": "0,10 % (minimum 1 €), demi-écart achat-vente, 2 pb de glissement, impact de marché, TTF et droit de timbre, change 0,25 % sur les lignes en devise.",
    "Banque de réseau": "0,50 % (minimum 10 €), change 0,50 %, et le reste comme le courtier en ligne.",
    "Institutionnel": "0,02 % sans minimum, 1 pb de glissement, impact de marché, taxes, change 0,02 %.",
}


# ----------------------------------------------------------------------------- config
@dataclass
class AdvancedConfig:
    initial_capital: float = 100_000.0
    rebalance: str | list = "monthly"
    execution_delay: int = 1
    lookback: int = 0
    drift_threshold: float = 0.0
    cash_rate: float = 0.0
    cash_prices: pd.Series | None = None
    costs: CostModel = field(default_factory=CostModel)
    leverage: float = 1.0              # the rule's weights are multiplied by this
    max_gross: float = 1.0             # sum of |weights| allowed (2.0 = up to 2x exposure)
    allow_short: bool = False
    debit_spread: float = 0.015        # borrowing at cash rate + 1.5 %
    borrow_fee: float = 0.005          # 0.5 % a year on the shares borrowed for a short
    short_rebate: bool = False         # short proceeds earn the cash rate
    cfd_overnight: float = 0.025       # 2.5 % a year on the CFD notional
    maintenance_margin: float = 0.0    # 0 = no margin call (e.g. 0.25 for a margin account)
    min_trade: float = 0.0             # order lines below this (EUR) are not sent
    management_fee: float = 0.0        # a year, on the portfolio value, taken every day
    performance_fee: float = 0.0       # on the gain above the high-water mark, taken at each year end
    triggers: tuple = ()               # Trigger(...): checked at every close between two scheduled decisions

    def __post_init__(self):
        if self.initial_capital <= 0:
            raise ValueError("le capital de départ doit être positif")
        if int(self.execution_delay) < 1:
            raise ValueError("execution_delay must be >= 1: same-bar execution is look-ahead")
        if self.leverage <= 0:
            raise ValueError("le levier doit être positif")
        if self.max_gross < 1.0 or self.max_gross > 10:
            raise ValueError("l'exposition brute autorisée doit être entre 1 et 10")
        if self.leverage > self.max_gross + 1e-12:
            raise ValueError(f"levier {self.leverage:g} supérieur à l'exposition autorisée {self.max_gross:g}")
        if not 0 <= self.maintenance_margin < 1:
            raise ValueError("la marge de maintenance doit être entre 0 et 1")
        if not 0 <= self.management_fee < 0.2 or not 0 <= self.performance_fee < 0.5:
            raise ValueError("frais de gestion entre 0 et 20 %, commission de performance entre 0 et 50 %")
        self.triggers = tuple(self.triggers)

    @property
    def fee_rate(self) -> float:                       # what the rest of the package reads
        return self.costs.commission_rate

    @property
    def borrows(self) -> bool:
        return self.max_gross > 1.0 or self.allow_short


@dataclass
class RunState:
    """Everything a run holds at the close of ``date``: enough to resume it later on new
    prices without replaying the past (``run_advanced(..., resume=state)``). Plain data:
    ``to_dict`` / ``from_dict`` store it as JSON."""
    date: pd.Timestamp
    cash: float
    units: dict
    pending: list                      # [(decision date, {ticker: weight}, kind)], not executed yet
    last_target: dict | None
    peak: float
    latched: bool
    hwm: float
    ruined: bool
    decided: bool
    totals: dict

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["date"] = str(pd.Timestamp(self.date).date())
        d["pending"] = [[str(pd.Timestamp(a).date()), w, k] for a, w, k in self.pending]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RunState":
        return cls(**{**d, "date": pd.Timestamp(d["date"]),
                      "pending": [(pd.Timestamp(a), w, k) for a, w, k in d["pending"]]})


@dataclass
class AdvancedResult:
    result: BacktestResult
    costs: dict                         # totals by kind, in euros
    events: pd.DataFrame                # margin calls and ruin
    ruined: bool = False
    state: RunState | None = None       # where the run stopped: resume from here

    @property
    def nav(self) -> pd.Series:
        return self.result.nav


COST_LABELS = {"commission": "Commissions", "spread": "Écart achat-vente", "slippage": "Glissement",
               "impact": "Impact de marché", "tax": "Taxes (TTF, droit de timbre)", "fx": "Change de devises",
               "management": "Frais de gestion", "performance": "Commission de performance",
               "debit": "Intérêts d'emprunt (levier)", "borrow": "Emprunt de titres (vente à découvert)",
               "cfd": "Financement des CFD", "credit": "Intérêts reçus sur les liquidités"}


# ---------------------------------------------------------------------------- triggers
TRIGGERS = {
    "stop": "Stop du portefeuille : tout en liquidités si la perte depuis le dernier sommet atteint le seuil",
    "derive": "Rééquilibrage anticipé si une ligne s'écarte de sa cible de plus que le seuil",
    "volatilite": "Exposition réduite si la volatilité des 20 dernières séances dépasse le seuil (annuel)",
}


@dataclass(frozen=True)
class Trigger:
    """Watched at every close between two scheduled decisions, on prices up to that close;
    its order is executed at the next close, like any decision. Once it has fired it waits
    for the next scheduled decision (no ping-pong)."""
    kind: str
    threshold: float

    def __post_init__(self):
        if self.kind not in TRIGGERS:
            raise ValueError(f"déclencheur inconnu : {self.kind!r}")
        if not 0 < self.threshold < 5:
            raise ValueError("seuil de déclencheur hors bornes")


# ------------------------------------------------------------------------------ rules
def leveraged(rule: WeightFn, factor: float) -> WeightFn:
    def decide(ctx: DecisionContext):
        w = rule(ctx)
        return None if w is None else pd.Series(w, dtype=float) * factor
    return decide


def long_short_momentum(lookback: int = 252, k: int = 3, skip: int = 21, gross: float = 1.0):
    """Long the ``k`` best performers over ``lookback`` bars (skipping the last ``skip``),
    short the ``k`` worst, half of ``gross`` on each side: a market-neutral bet."""
    def decide(ctx: DecisionContext):
        h = ctx.history
        if len(h) <= lookback:
            return None
        past = h.iloc[-1 - skip] / h.iloc[-1 - lookback] - 1
        ranked = past.sort_values()
        n = min(k, len(ranked) // 2)
        if n == 0:
            return None
        w = pd.Series(0.0, index=h.columns)
        w[ranked.index[-n:]] = gross / 2 / n
        w[ranked.index[:n]] = -gross / 2 / n
        return w
    return decide


def trend_long_short(days: int = 200, gross: float = 1.0):
    """Each asset long above its ``days``-bar average, short below; equal gross per line."""
    def decide(ctx: DecisionContext):
        h = ctx.history
        if len(h) <= days:
            return None
        sign = np.sign(h.iloc[-1] - h.iloc[-days:].mean())
        return sign * gross / len(sign)
    return decide


# ----------------------------------------------------------------------------- engine
def _targets(raw, tickers: pd.Index, config: AdvancedConfig) -> pd.Series:
    w = pd.Series(raw, dtype=np.float64)
    w.index = [str(t).strip().upper() for t in w.index]
    unknown = sorted(set(w.index) - set(tickers))
    if unknown:
        raise ValueError(f"weights reference unknown tickers {unknown}")
    if not np.isfinite(w.to_numpy()).all():
        raise ValueError("weights must be finite")
    w = w.reindex(tickers, fill_value=0.0) * config.leverage
    if not config.allow_short and (w < -1e-12).any():
        raise ValueError("vente à découvert non autorisée : la règle a donné des poids négatifs")
    gross = float(w.abs().sum())
    if gross > config.max_gross + 1e-9:
        raise ValueError(f"exposition brute {gross:.4f} > {config.max_gross:g} autorisée")
    return w if config.allow_short else w.clip(lower=0.0)


def run_advanced(prices: pd.DataFrame, weight_fn: WeightFn, config: AdvancedConfig | None = None,
                 instruments: Mapping[str, Instrument | str] | None = None,
                 label: str = "strategy", resume: RunState | None = None) -> AdvancedResult:
    """``resume``: continue a run stopped at ``resume.date`` (its ``state``) on a panel that
    holds the same past plus new days; the result covers the new days only, and is the
    same as the full run's on those days."""
    config = config or AdvancedConfig()
    prices = validate_prices(prices)
    if prices.isna().any().any():
        raise ValueError("the backtest needs a full panel; use data.common_window() first")
    index, tickers = prices.index, prices.columns
    given = dict(instruments or {})
    instr = []
    for t in tickers:
        i = given.get(t)
        instr.append(i if isinstance(i, Instrument) else guess_instrument(t, i))
    is_cfd = np.array([i.kind == "cfd" for i in instr])
    px = prices.to_numpy()
    n_bars, n_assets = px.shape
    sigma = prices.pct_change().rolling(20, min_periods=5).std().shift(1).to_numpy()

    decide_at = set(decision_dates(index, config.rebalance, config.lookback))
    exec_at: dict[int, list[int]] = {}
    for d in decide_at:
        e = d + int(config.execution_delay)
        if e < n_bars:
            exec_at.setdefault(e, []).append(d)

    cash = np.empty(n_bars)
    invested = np.empty(n_bars)
    units_hist = np.zeros((n_bars, n_assets))
    units = np.zeros(n_assets)
    cash_now = float(config.initial_capital)
    pending: dict[int, pd.Series] = {}
    trades, decisions, events, notes = [], [], [], []
    totals = {k: 0.0 for k in COST_LABELS}
    costs_bar = np.zeros(n_bars)
    carry_bar = np.zeros(n_bars)
    day_gap = np.r_[0.0, np.diff(index.to_numpy(dtype="datetime64[ns]")).astype("int64") / 86_400e9]
    cash_growth = None
    if config.cash_prices is not None:
        level = config.cash_prices.sort_index()
        level = level[~level.index.duplicated(keep="last")].reindex(index, method="ffill")
        cash_growth = (level / level.shift(1)).fillna(1.0).to_numpy()
    ruined = False
    hwm = float(config.initial_capital)
    kinds: dict[int, str] = {}                 # decision bar -> "prévu" or the trigger that fired
    last_target, peak, latched = None, float(config.initial_capital), False
    t0, decided = 0, False
    if resume is not None:
        if resume.date not in index:
            raise ValueError(f"la date de reprise {pd.Timestamp(resume.date).date()} n'est pas dans les cours")
        t0 = int(index.get_loc(resume.date)) + 1
        if t0 >= n_bars:
            raise ValueError("aucune séance nouvelle après la date de reprise")
        cash_now = float(resume.cash)
        units = np.array([float(resume.units.get(t, 0.0)) for t in tickers])
        for when, weights, kind in resume.pending:
            d = int(index.get_loc(pd.Timestamp(when)))
            pending[d] = pd.Series(weights, dtype=float).reindex(tickers, fill_value=0.0)
            kinds[d] = kind
            e = d + int(config.execution_delay)
            if d not in decide_at and e < n_bars:
                exec_at.setdefault(e, []).append(d)
        last_target = (pd.Series(resume.last_target, dtype=float).reindex(tickers, fill_value=0.0).to_numpy()
                       if resume.last_target is not None else None)
        peak, latched, hwm, ruined, decided = resume.peak, resume.latched, resume.hwm, resume.ruined, resume.decided
        totals.update(resume.totals)

    def execute(t: int, w: np.ndarray, forced: str = "") -> None:
        nonlocal cash_now, units
        value_now = units * px[t]
        nav_pre = cash_now + value_now.sum()

        def orders(nav_post):
            n = w * nav_post - value_now
            if config.min_trade and not forced:
                n = np.where(np.abs(n) < config.min_trade, 0.0, n)
            return n

        nav_post = nav_pre
        for _ in range(50):
            parts = config.costs.line_costs(orders(nav_post), instr, sigma[t], index[t])
            new = nav_pre - sum(p.sum() for p in parts.values())
            if abs(new - nav_post) < 1e-12 * max(abs(nav_pre), 1.0):
                nav_post = new
                break
            nav_post = new
        notional = orders(nav_post)
        parts = config.costs.line_costs(notional, instr, sigma[t], index[t])
        per_line = sum(parts.values())
        cost = float(per_line.sum())
        delta = notional / px[t]
        units = units + delta
        cash_now = cash_now - notional.sum() - cost
        costs_bar[t] += cost
        for k, v in parts.items():
            totals[k] += float(v.sum())
        for j in np.flatnonzero(np.abs(notional) > 1e-9):
            trades.append({"date": index[t], "ticker": tickers[j], "units": delta[j], "price": px[t, j],
                           "notional": notional[j], "fee": per_line[j],
                           **{k: v[j] for k, v in parts.items()}, "forcé": forced})
        if not config.borrows and cash_now < -1e-6 * max(nav_pre, 1.0):
            raise BacktestError(f"cash went negative ({cash_now:.4f}) on {index[t].date()}")

    def decide_step(t: int, nav_t: float) -> None:
        """Steps 5 and 6 at the close of bar t (history up to t only)."""
        nonlocal latched
        if t + config.execution_delay >= n_bars or ruined:
            return
        placed = False
        if t in decide_at:
            current = pd.Series(units * px[t] / nav_t if nav_t > 0 else 0.0, index=tickers)
            ctx = DecisionContext(date=index[t], history=prices.iloc[: t + 1], current_weights=current, nav=nav_t)
            raw = weight_fn(ctx)
            if raw is not None:
                target = _targets(raw, tickers, config)
                drift = float((target - current).abs().max())
                if not (decisions or decided) or drift >= config.drift_threshold:
                    pending[t], kinds[t], placed = target, "prévu", True
                    decisions.append({"decision_date": index[t], "execution_date": index[t + config.execution_delay],
                                      "max_drift": drift, "motif": "prévu", **{f"w_{k}": v for k, v in target.items()}})
        # 6. triggers, between two scheduled decisions, once each
        if config.triggers and not placed and not pending and not latched and last_target is not None and nav_t > 0:
            weights_now = units * px[t] / nav_t
            for trig in config.triggers:
                new, why = None, ""
                if trig.kind == "stop" and nav_t / peak - 1 <= -trig.threshold:
                    new, why = np.zeros(n_assets), f"perte de {1 - nav_t / peak:.1%} depuis le dernier sommet"
                elif trig.kind == "derive" and np.abs(weights_now - last_target).max() >= trig.threshold:
                    new, why = last_target.copy(), f"écart de {np.abs(weights_now - last_target).max():.1%} à la cible"
                elif trig.kind == "volatilite" and t >= 21:
                    # the risk of what is held now, on its last 20 sessions (no stored history needed)
                    window = px[t - 20: t + 1]
                    vol = float(np.std((window[1:] / window[:-1] - 1) @ weights_now, ddof=1) * np.sqrt(252))
                    if vol >= trig.threshold:
                        new, why = last_target * trig.threshold / vol, f"volatilité de {vol:.1%} sur 20 séances"
                if new is not None:
                    target = pd.Series(new, index=tickers)
                    pending[t], kinds[t], latched = target, trig.kind, True
                    exec_at.setdefault(t + int(config.execution_delay), []).append(t)
                    decisions.append({"decision_date": index[t], "execution_date": index[t + config.execution_delay],
                                      "max_drift": float(np.abs(new - weights_now).max()), "motif": trig.kind,
                                      **{f"w_{k}": v for k, v in target.items()}})
                    events.append({"date": index[t], "événement": "déclencheur : " + trig.kind, "détail": why})
                    break


    if t0:                                     # the saved close could not decide: no next day then
        decide_step(t0 - 1, resume.cash + float((units * px[t0 - 1]).sum()))

    for t in range(t0, n_bars):
        # 1. carry since the previous bar, on what was held over it
        if t > 0:
            years = day_gap[t] / 365.0
            held = units * px[t - 1]
            short_value = float(-held[held < 0].sum())
            if cash_growth is not None:
                rate_growth = cash_growth[t] - 1.0
            else:
                rate_growth = (1.0 + config.cash_rate) ** years - 1.0 if config.cash_rate else 0.0
            earning = max(cash_now - (0.0 if config.short_rebate else short_value), 0.0)
            credit = earning * rate_growth
            debit = max(-cash_now, 0.0) * (rate_growth + (1.0 + config.debit_spread) ** years - 1.0)
            borrow = short_value * ((1.0 + config.borrow_fee) ** years - 1.0)
            cfd = float(np.abs(held[is_cfd]).sum()) * ((1.0 + config.cfd_overnight) ** years - 1.0)
            if cash_growth is None and not config.cash_rate:
                credit = 0.0
            carry = credit - debit - borrow - cfd
            cash_now += carry
            carry_bar[t] = carry
            totals["credit"] += credit
            totals["debit"] += debit
            totals["borrow"] += borrow
            totals["cfd"] += cfd
            if config.management_fee:
                # taken pro rata from every line, like a fund's fees: no cash is borrowed to pay them
                frac = 1.0 - (1.0 - config.management_fee) ** years
                nav_now = cash_now + float((units * px[t]).sum())
                if nav_now > 0:
                    fee = frac * nav_now
                    units = units * (1.0 - frac)
                    cash_now *= (1.0 - frac)
                    carry_bar[t] -= fee
                    totals["management"] += fee

        # 2. execute orders decided earlier (the latest decision wins)
        live = [d for d in exec_at.get(t, ()) if d in pending]
        if live and not ruined:
            chosen = max(live)
            target = pending.pop(chosen)
            for older in live:
                pending.pop(older, None)
            execute(t, target.to_numpy())
            if kinds.get(chosen, "prévu") == "prévu":
                last_target, latched = target.to_numpy(), False
                peak = cash_now + float((units * px[t]).sum())

        # 3. margin: equity against gross exposure at this close
        value = units * px[t]
        gross = float(np.abs(value).sum())
        equity = cash_now + float(value.sum())
        if gross > 1e-9 and not ruined:
            if equity <= 0:
                execute(t, np.zeros(n_assets), forced="ruine")
                ruined = True
                pending.clear()
                events.append({"date": index[t], "événement": "ruine",
                               "détail": "les pertes ont absorbé toute la mise : tout est soldé"})
            elif config.maintenance_margin and equity / gross < config.maintenance_margin:
                scale = equity * config.max_gross / gross * 0.999
                execute(t, value / equity * scale if equity > 0 else np.zeros(n_assets),
                        forced="appel de marge")
                events.append({"date": index[t], "événement": "appel de marge",
                               "détail": f"marge {equity / gross:.1%} sous {config.maintenance_margin:.0%} : "
                                         f"positions réduites"})

        # 3b. performance fee, taken at each year end on the gain above the high-water mark
        year_end = (index[t].year != index[t + 1].year if t < n_bars - 1
                    else (index[t] + pd.offsets.BDay(1)).year != index[t].year)
        if config.performance_fee and year_end:
            nav_now = cash_now + float((units * px[t]).sum())
            if nav_now > hwm:
                fee = config.performance_fee * (nav_now - hwm)
                units = units * (1.0 - fee / nav_now)
                cash_now *= (1.0 - fee / nav_now)
                costs_bar[t] += fee
                totals["performance"] += fee
                hwm = nav_now - fee

        # 4. book the bar
        units_hist[t] = units
        invested[t] = float((units * px[t]).sum())
        cash[t] = cash_now
        nav_t = cash[t] + invested[t]
        if not np.isfinite(nav_t):
            raise BacktestError(f"non-finite NAV on {index[t].date()}")

        peak = max(peak, nav_t)
        decide_step(t, nav_t)

    state = RunState(date=index[-1], cash=float(cash_now), units={k: float(v) for k, v in zip(tickers, units)},
                     pending=[(index[d], {k: float(v) for k, v in w.items()}, kinds.get(d, "prévu"))
                              for d, w in sorted(pending.items())],
                     last_target=(None if last_target is None else {k: float(v) for k, v in zip(tickers, last_target)}),
                     peak=float(peak), latched=bool(latched), hwm=float(hwm), ruined=bool(ruined),
                     decided=bool(decisions) or decided, totals=dict(totals))
    if t0:                                     # a resumed run reports its new days only
        before = np.array([float(resume.units.get(k, 0.0)) for k in tickers])
        start_nav = resume.cash + float((before * px[t0 - 1]).sum())
        expected = start_nav + float((before * (px[t0] - px[t0 - 1])).sum()) + carry_bar[t0] - costs_bar[t0]
        if abs(cash[t0] + invested[t0] - expected) > IDENTITY_TOLERANCE * max(config.initial_capital, abs(start_nav)):
            raise BacktestError("the resumed run does not start from the saved state")
        index, px = index[t0:], px[t0:]
        cash, invested, units_hist = cash[t0:], invested[t0:], units_hist[t0:]
        carry_bar, costs_bar = carry_bar[t0:], costs_bar[t0:]
        prices = prices.iloc[t0:]
    nav = cash + invested
    if len(nav) > 1:
        explained = (units_hist[:-1] * (px[1:] - px[:-1])).sum(axis=1) + carry_bar[1:] - costs_bar[1:]
        gap = np.abs(np.diff(nav) - explained)
        worst = int(np.argmax(gap))
        if gap[worst] > IDENTITY_TOLERANCE * max(config.initial_capital, float(np.max(np.abs(nav)))):
            raise BacktestError(f"P&L identity broken on {index[worst + 1].date()}: unexplained {gap[worst]:.3e}")
    if not t0:
        first_gap = abs(nav[0] - (config.initial_capital + carry_bar[0] - costs_bar[0]))
        if first_gap > IDENTITY_TOLERANCE * config.initial_capital:
            raise BacktestError(f"first bar does not start from the initial capital ({first_gap:.3e})")
    if not decisions and not decided:
        notes.append("the weighting rule never returned weights: the run stayed in cash")

    nav_s = pd.Series(nav, index=index, name="nav")
    columns = ["date", "ticker", "units", "price", "notional", "fee", "commission", "spread", "slippage", "impact",
               "tax", "fx", "forcé"]
    result = BacktestResult(
        nav=nav_s, cash=pd.Series(cash, index=index, name="cash"),
        invested=pd.Series(invested, index=index, name="invested"),
        weights=pd.DataFrame(units_hist * px, index=index, columns=tickers).div(nav_s, axis=0),
        units=pd.DataFrame(units_hist, index=index, columns=tickers),
        trades=pd.DataFrame(trades, columns=columns if not trades else None),
        decisions=pd.DataFrame(decisions), config=config, label=label,
        fees_paid=float(costs_bar.sum()), notes=notes, prices=prices)
    return AdvancedResult(result=result, costs=totals,
                          events=pd.DataFrame(events, columns=["date", "événement", "détail"]), ruined=ruined,
                          state=state)


def cost_table(res: AdvancedResult) -> pd.DataFrame:
    """Every cost, in euros and in % of the starting capital (a positive line is a cost)."""
    cap = res.result.config.initial_capital
    rows = []
    for k, label in COST_LABELS.items():
        v = res.costs.get(k, 0.0)
        signed = -v if k == "credit" else v
        if abs(v) > 1e-9:
            rows.append({"poste": label, "euros": signed, "% du capital": signed / cap})
    return pd.DataFrame(rows, columns=["poste", "euros", "% du capital"])
