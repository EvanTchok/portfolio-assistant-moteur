"""Croisement de moyennes dans Portfolio Assistant, vectorbt et NautilusTrader.
pip install vectorbt nautilus_trader, puis : python vitesse/trading.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
"""Same moving-average crossover in Portfolio Assistant (algo mode), vectorbt and NautilusTrader."""
import json, time, platform, statistics, warnings
from collections import deque
from decimal import Decimal
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import vectorbt as vbt
from portfolio_assistant.algo import AlgoSpec, run_algo, signals
from portfolio_assistant.execution import CostModel

N, FAST, SLOW, CAP = 50_000, 20, 100, 100_000.0
rng = np.random.default_rng(7)
idx = pd.date_range("2000-01-03", periods=N, freq="h")
close = np.round(100 * np.cumprod(1 + rng.normal(0, 0.004, N)), 2)
prices = pd.DataFrame({"X": close}, index=idx)
spec = AlgoSpec(signal="croisement", fast=FAST, slow=SLOW, size=1.0, max_positions=1, stop_loss=None,
                costs=CostModel(0.0), initial_capital=CAP)

def ours():
    return run_algo(prices, spec)

sig = signals(prices, spec)
ent, ex_ = sig["entry_long"]["X"], sig["exit_long"]["X"]
def vb():
    # decided at the close, executed at the next close
    pf = vbt.Portfolio.from_signals(prices["X"], entries=ent.shift(1, fill_value=False), exits=ex_.shift(1, fill_value=False),
                                    init_cash=CAP, fees=0.0, size=np.inf, freq="1h")
    return pf

# ---------------------------------------------------------------- NautilusTrader
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, StrategyConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.wranglers import BarDataWrangler
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from nautilus_trader.trading.strategy import Strategy

class Cfg(StrategyConfig, frozen=True):
    bar_type: str
    fast: int = FAST
    slow: int = SLOW

class Cross(Strategy):
    def __init__(self, config):
        super().__init__(config)
        self.bt = BarType.from_str(config.bar_type)
        self.win = deque(maxlen=config.slow)
        self.prev = None
        self.todo = None
    def on_start(self):
        self.instrument = self.cache.instrument(self.bt.instrument_id)
        self.subscribe_bars(self.bt)
    def on_bar(self, bar):
        c = float(bar.close)
        # an order decided at the previous close goes out now, so it fills at THIS close
        if self.todo == "buy" and self.portfolio.is_flat(self.bt.instrument_id):
            cash = float(self.portfolio.account(self.instrument.id.venue).balance_free(USD).as_double())
            qty = int(cash // c)                 # whole shares: the one difference with the two others
            if qty > 0:
                self.submit_order(self.order_factory.market(self.bt.instrument_id, OrderSide.BUY,
                                                            self.instrument.make_qty(qty)))
        elif self.todo == "sell" and not self.portfolio.is_flat(self.bt.instrument_id):
            self.close_all_positions(self.bt.instrument_id)
        self.todo = None
        self.win.append(c)
        if len(self.win) < self.config.slow:
            return
        w = list(self.win)
        diff = sum(w[-self.config.fast:]) / self.config.fast - sum(w) / self.config.slow
        prev, self.prev = self.prev, diff
        if prev is None:
            return
        if diff > 0 and prev <= 0:
            self.todo = "buy"
        elif diff < 0 and prev >= 0:
            self.todo = "sell"

def nautilus():
    eng = BacktestEngine(BacktestEngineConfig(logging=LoggingConfig(bypass_logging=True)))
    venue = Venue("XNAS")
    eng.add_venue(venue=venue, oms_type=OmsType.NETTING, account_type=AccountType.CASH, base_currency=USD,
                  starting_balances=[Money(CAP, USD)])
    inst = TestInstrumentProvider.equity(symbol="X", venue="XNAS")
    eng.add_instrument(inst)
    bt = f"{inst.id}-1-HOUR-LAST-EXTERNAL"
    from nautilus_trader.model.data import Bar
    btype = BarType.from_str(bt)
    vol = inst.make_qty(1_000_000_000)
    ts = (idx.as_unit("ns").asi8 + 3_600_000_000_000).tolist()        # a bar is stamped at its close
    bars = []
    for c, s in zip(close.tolist(), ts):
        p = inst.make_price(c)
        bars.append(Bar(btype, p, p, p, p, vol, s, s))
    eng.add_data(bars)
    eng.add_strategy(Cross(Cfg(bar_type=bt)))
    eng.run()
    acct = eng.portfolio.account(venue)
    final = float(acct.balance_total(USD).as_double())
    pos = eng.cache.positions()
    last = close[-1]
    open_val = sum(float(p.quantity) * last for p in pos if p.is_open)
    trades = (len([o for o in eng.cache.orders() if o.is_closed]) + 1) // 2
    eng.dispose()
    return final + open_val, trades

def timeit(f, k=3):
    out = []
    for _ in range(k):
        t = time.perf_counter(); r = f(); out.append(time.perf_counter() - t)
    return statistics.median(out), r

t_first = time.perf_counter(); vb(); vb_first = time.perf_counter() - t_first
t_o, r_o = timeit(ours)
t_v, r_v = timeit(vb)
t_n, r_n = timeit(nautilus, k=1)
v_o = float(r_o.nav.iloc[-1]); v_v = float(r_v.value().iloc[-1]); v_n = r_n[0]
res = {"barres": N, "Portfolio Assistant (mode algorithmique)": {"secondes": t_o, "valeur finale": v_o, "positions": int(len(r_o.journal))},
       "vectorbt": {"secondes": t_v, "première fois": vb_first, "valeur finale": v_v, "positions": int(r_v.trades.count())},
       "NautilusTrader": {"secondes": t_n, "valeur finale": v_n, "positions": int(r_n[1])},
       "environnement": {"python": platform.python_version(), "vectorbt": vbt.__version__,
                         "nautilus_trader": __import__("nautilus_trader").__version__}}
print(json.dumps(res, indent=1, ensure_ascii=False))
json.dump(res, open(pathlib.Path(__file__).with_name("trading.json"), "w"), ensure_ascii=False, indent=1)
