"""Vitesse du moteur face à vectorbt, sur la même stratégie (parts égales, rééquilibrage mensuel,
0,10 % de frais, exécution le lendemain). pip install vectorbt, puis : python vitesse/bench.py
Résultats publiés : vitesse/resultats.json."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json, time, platform, statistics
import numpy as np, pandas as pd
import vectorbt as vbt
from portfolio_assistant.engine import run_backtest, BacktestConfig, decision_dates
from portfolio_assistant.strategies import equal_weight
from portfolio_assistant.execution import run_advanced, AdvancedConfig

def panel(n_bars, n_assets, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2005-01-03", periods=n_bars)
    return pd.DataFrame(100*np.cumprod(1+rng.normal(3e-4, 0.011, (n_bars, n_assets)), axis=0), index=idx,
                        columns=[f"A{i:02d}" for i in range(n_assets)])

def ours(p):
    return run_backtest(p, equal_weight(), BacktestConfig(rebalance="monthly", lookback=0, fee_rate=0.001)).nav

def ours_ref(p):
    return run_backtest(p, equal_weight(), BacktestConfig(rebalance="monthly", lookback=0, fee_rate=0.001),
                        reference=True).nav

def ours_adv(p):
    return run_advanced(p, equal_weight(), AdvancedConfig(rebalance="monthly", lookback=0)).nav

def vbt_run(p):
    exec_bars = [d + 1 for d in decision_dates(p.index, "monthly", 0)]
    size = pd.DataFrame(np.nan, index=p.index, columns=p.columns)
    size.iloc[exec_bars] = 1.0 / p.shape[1]
    pf = vbt.Portfolio.from_orders(p, size=size, size_type="targetpercent", group_by=True, cash_sharing=True,
                                   call_seq="auto", fees=0.001, init_cash=100_000.0, freq="1D")
    return pf.value()

def timeit(f, p, k=5):
    out = []
    for _ in range(k):
        t = time.perf_counter(); f(p); out.append(time.perf_counter() - t)
    return statistics.median(out)

rows = []
for n_bars, n_assets in [(5000, 10), (5000, 50)]:
    p = panel(n_bars, n_assets)
    t0 = time.perf_counter(); v = vbt_run(p); first = time.perf_counter() - t0
    a, b, c = ours(p), ours_adv(p), vbt_run(p)
    gap = float(abs(a.iloc[-1] / c.iloc[-1] - 1))
    rows.append({"séances": n_bars, "lignes": n_assets,
                 "Portfolio Assistant (moteur du laboratoire)": timeit(ours, p),
                 "Portfolio Assistant (boucle de référence)": timeit(ours_ref, p),
                 "Portfolio Assistant (atelier)": timeit(ours_adv, p),
                 "vectorbt": timeit(vbt_run, p), "vectorbt, 1re exécution (compilation)": first,
                 "écart de valeur finale": gap, "écart atelier / laboratoire": float(abs(a.iloc[-1] / b.iloc[-1] - 1)),
                 "écart chemin rapide / boucle de référence": float(abs(a.iloc[-1] / ours_ref(p).iloc[-1] - 1))})
info = {"python": platform.python_version(), "vectorbt": vbt.__version__, "pandas": pd.__version__,
        "machine": platform.machine(), "processeur": platform.processor() or "inconnu"}
print(json.dumps({"lignes": rows, "environnement": info}, indent=1, ensure_ascii=False))
json.dump({"lignes": rows, "environnement": info}, open(pathlib.Path(__file__).with_name("resultats.json"), "w"), ensure_ascii=False, indent=1)
