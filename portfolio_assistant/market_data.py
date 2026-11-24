"""Real market prices from Yahoo Finance, with a local cache.

* Prices are ADJUSTED (splits and dividends folded in), so a backtest on them is a
  total-return backtest: a dividend shows up as performance, as it does for the investor.
* Each ticker is cached in ``data/cache/<TICKER>.csv``. A second call reads the disk;
  a later call only downloads the days that are missing at the end.
* Tickers quoted on different exchanges do not share holidays. Dates where one of them
  has no price are DROPPED and counted, never filled: a filled price is an invented one.
* Today's bar is never used: before the close it is a live price, not a closing price.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .data import DataError, validate_prices

DEFAULT_CACHE = Path("donnees_cabinet") / "cache"        # see chemins.py


def _yahoo_download(ticker: str, start: str | None, end: str | None) -> pd.Series:
    """One ticker's adjusted close from Yahoo Finance (needs internet and yfinance)."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on the machine
        raise DataError("yfinance is not installed: run `pip install yfinance`") from exc
    frame = yf.download(ticker, start=start, end=end, auto_adjust=True,
                        progress=False, multi_level_index=False)
    if frame is None or frame.empty or "Close" not in frame.columns:
        return pd.Series(dtype="float64", name=ticker)
    close = frame["Close"]
    if isinstance(close, pd.DataFrame):  # older yfinance versions
        close = close.iloc[:, 0]
    return final_closes(close.rename(ticker))


def _download(ticker: str, start: str | None, end: str | None) -> pd.Series:
    """The source chosen in the settings (Yahoo by default; see sources_prix.py)."""
    from . import sources_prix
    return sources_prix.download(ticker, start, end)


def final_closes(series: pd.Series, now: pd.Timestamp | None = None) -> pd.Series:
    """Drops today's bar. During the session Yahoo returns TODAY with the live price, not a
    close; stored as a close it would be compared, tracked and logged as if final. A day's
    price is used from the next day on (Paris time), when every market of the day is closed."""
    if series.empty:
        return series
    idx = pd.DatetimeIndex(series.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    series = series.copy()
    series.index = idx.normalize()
    now = now if now is not None else pd.Timestamp.now(tz="Europe/Paris")
    today = (now.tz_convert("Europe/Paris").tz_localize(None) if now.tzinfo else now).normalize()
    return series[series.index < today]


def _cache_file(cache_dir: Path, ticker: str) -> Path:
    safe = ticker.replace("/", "_").replace("^", "IDX_")
    return cache_dir / f"{safe}.csv"


def _read_cache(path: Path) -> pd.Series | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    if frame.empty:
        return None
    return frame.iloc[:, 0].astype("float64")


def fetch_ticker(ticker: str, start: str = "2005-01-01", *, cache_dir: str | Path = DEFAULT_CACHE,
                 refresh: bool = True, downloader: Callable = _download) -> pd.Series:
    """Adjusted daily closes for one ticker, cached on disk.

    ``refresh=True`` tops the cache up with the days it lacks (after its last date, or a day
    Yahoo had skipped among the last ten). Adjusted
    prices are rewritten backwards by Yahoo after each dividend, so a top-up also
    re-downloads the last 10 days and, if they no longer match, the whole history.
    """
    ticker = ticker.strip().upper()
    if downloader is _download:              # the cabinet's own imported prices come first (never cached)
        from . import sources_prix
        own = sources_prix.cours_importe(ticker)
        if own is not None and len(own):
            return own.loc[start:].rename(ticker)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_file(cache_dir, ticker)
    cached = _read_cache(path)

    if cached is not None and cached.index.min() <= pd.Timestamp(start) + pd.Timedelta(days=7):
        if not refresh:
            return cached.loc[start:].rename(ticker)
        overlap_start = cached.index[max(len(cached) - 10, 0)]
        recent = downloader(ticker, overlap_start.strftime("%Y-%m-%d"), None).dropna()
        common = cached.index.intersection(recent.index)
        drifted = len(common) and (
            (recent.loc[common] / cached.loc[common] - 1.0).abs().max() > 1e-6)
        if drifted:  # a dividend or split rewrote the adjusted history
            series = downloader(ticker, start, None).dropna()
        else:
            # every date Yahoo returns that the cache lacks is added, not only the later ones:
            # Yahoo sometimes skips a day (often the day before, during a session) and sends
            # it again later; appending only after the last date would leave a hole forever
            series = pd.concat([cached, recent[~recent.index.isin(cached.index)]]).sort_index()
    else:
        series = downloader(ticker, start, None).dropna()
        if series.empty and cached is not None:
            # a fund younger than ``start`` always looks incomplete; when the source is down
            # (or refuses this server) the prices already on disk are better than none
            series = cached

    if series.empty:
        raise DataError(f"no price returned for {ticker}: check the Yahoo ticker "
                        "(Euronext Paris ends with .PA, Xetra with .DE)")
    series = series[~series.index.duplicated(keep="last")].sort_index().astype("float64")
    series.index = pd.DatetimeIndex(series.index).tz_localize(None).normalize()
    series = final_closes(series)          # also cleans a live price stored by an older version
    if series.empty:
        raise DataError(f"no closing price yet for {ticker}")
    series.rename(ticker).to_frame().to_csv(path)
    return series.loc[start:].rename(ticker)


def fetch_prices(tickers: Iterable[str], start: str = "2005-01-01", end: str | None = None, *,
                 cache_dir: str | Path = DEFAULT_CACHE, refresh: bool = True,
                 downloader: Callable = _download, verbose: bool = True,
                 currencies: dict[str, str] | None = None, base_currency: str | None = None
                 ) -> pd.DataFrame:
    """A clean price panel ready for ``run_backtest``.

    The panel starts when the YOUNGEST ticker starts (every asset must exist to be
    allocated) and keeps only dates where every ticker has a price.

    ``currencies`` ({ticker: "USD", ...}) and ``base_currency`` ("EUR") express every
    price in the investor's currency: a USD price is divided by the EURUSD rate of the
    same day (Yahoo "EURUSD=X", dollars per euro). A day without a rate is dropped like a
    day without a price: rates are never filled either.
    """
    tickers = [t.strip().upper() for t in tickers]
    if len(set(tickers)) != len(tickers):
        raise DataError("a ticker is listed twice")
    columns, failed = {}, {}
    for ticker in tickers:
        try:
            columns[ticker] = fetch_ticker(ticker, start, cache_dir=cache_dir,
                                           refresh=refresh, downloader=downloader)
        except DataError as exc:
            failed[ticker] = str(exc)
    if failed:
        raise DataError("could not load: " + "; ".join(f"{t} ({m})" for t, m in failed.items()))

    raw = pd.DataFrame(columns)
    converted = []
    if currencies and base_currency:
        base_currency = base_currency.upper()
        for ticker in tickers:
            ccy = currencies.get(ticker, base_currency).upper()
            if ccy == base_currency:
                continue
            pair = f"{base_currency}{ccy}=X"                      # units of ccy per 1 base
            rate = fetch_ticker(pair, start, cache_dir=cache_dir, refresh=refresh,
                                downloader=downloader)
            raw[ticker] = raw[ticker] / rate.reindex(raw.index)    # missing rate -> NaN -> dropped
            converted.append(f"{ticker} ({ccy})")
    if end is not None:
        raw = raw.loc[:end]
    first = {t: s.first_valid_index() for t, s in raw.items()}
    common_start = max(first.values())
    window = raw.loc[common_start:]
    complete = window.dropna(how="any")
    dropped = len(window) - len(complete)

    if verbose:
        youngest = max(first, key=first.get)
        print(f"{len(tickers)} tickers, {complete.index[0].date()} -> {complete.index[-1].date()} "
              f"({len(complete)} days). Start set by {youngest}.")
        if converted:
            print(f"converted to {base_currency}: {', '.join(converted)}")
        if dropped:
            print(f"{dropped} dates dropped where at least one market was closed "
                  f"({dropped / max(len(window), 1):.1%} of the window).")
    if len(complete) < 2:
        raise DataError("fewer than two dates where every ticker quotes")
    return validate_prices(complete)


# ------------------------------------------------------------------- the session's range
def _yahoo_ranges(ticker: str, start: str | None) -> pd.DataFrame:
    """Open, high and low as fractions of the same day's close (adjusted like the close)."""
    import yfinance as yf
    frame = yf.download(ticker, start=start, auto_adjust=True, progress=False, multi_level_index=False)
    if frame is None or frame.empty or not {"Open", "High", "Low", "Close"} <= set(frame.columns):
        return pd.DataFrame(columns=["o", "h", "l"])
    out = pd.DataFrame({"o": frame["Open"] / frame["Close"], "h": frame["High"] / frame["Close"],
                        "l": frame["Low"] / frame["Close"]})
    out.index = pd.DatetimeIndex(out.index).tz_localize(None).normalize() if out.index.tz is not None \
        else pd.DatetimeIndex(out.index).normalize()
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def fetch_ranges(closes: pd.DataFrame, *, cache_dir: str | Path = DEFAULT_CACHE,
                 downloader: Callable | None = None) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """The day's open / high / low for a panel of closes, in the panel's own currency: the
    source gives them as fractions of its close, applied to the panel's close (so a price
    converted to euros keeps its range). Returns the frames and the tickers without data;
    a missing day stays missing (the engine then uses the close)."""
    folder = Path(cache_dir) / "seances"
    folder.mkdir(parents=True, exist_ok=True)
    get = downloader or _yahoo_ranges
    ratios, missing = {}, []
    for ticker in closes.columns:
        path = folder / f"{ticker.replace('/', '_').replace('^', 'IDX_')}.csv"
        held = pd.read_csv(path, index_col=0, parse_dates=True) if path.exists() else None
        if held is None or held.empty or held.index.max() < closes.index.max() - pd.Timedelta(days=7):
            try:
                fresh = get(ticker, str(closes.index.min().date()))
                if not fresh.empty:
                    held = fresh if held is None else pd.concat([held, fresh[~fresh.index.isin(held.index)]]).sort_index()
                    held.to_csv(path)
            except Exception:                    # no network, no yfinance, refused: keep what is on disk
                pass
        if held is None or held.empty:
            missing.append(ticker)
            continue
        ratios[ticker] = held.reindex(closes.index)
    frames = {}
    for key, col in (("open", "o"), ("high", "h"), ("low", "l")):
        frames[key] = pd.DataFrame({t: closes[t] * r[col] for t, r in ratios.items()}, index=closes.index) \
            .reindex(columns=closes.columns)
    # a range must contain its close; anything else is a bad row from the source: dropped
    bad = (frames["high"] < closes - 1e-9) | (frames["low"] > closes + 1e-9) | (frames["low"] > frames["high"])
    for key in frames:
        frames[key] = frames[key].mask(bad)
    return frames, missing
