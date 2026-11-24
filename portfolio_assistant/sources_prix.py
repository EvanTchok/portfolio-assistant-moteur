"""Where the prices come from: one setting, so that a licensed provider can replace Yahoo
without touching the rest of the software.

    PA_SOURCE_PRIX      « yahoo » (default) or « eodhd »
    PA_EODHD_CLE        the EODHD API key (only read from the environment, never stored)
    PA_SOURCE_CHANGE    « yahoo » or « bce » ; default: « bce » when the prices are licensed, else « yahoo »

Yahoo Finance is free but its terms exclude commercial use: fine for research and for free
trials, not for a paid service. EODHD (eodhd.com) sells end-of-day prices with a commercial
licence. The euro reference rates of the European Central Bank are free to reuse, commercial
use included, with the source named (« Source : statistiques de la BCE »).

Nothing changes by default: the lab keeps the prices and exchange rates it was built on.
Switching source changes the converted prices very slightly (the ECB rate is fixed around
14:15, Yahoo's is a closing rate), and the cache rebuilds itself on the first run.

Every downloader has the same shape as market_data._yahoo_download:
(ticker, start, end) -> adjusted daily closes, today's bar excluded.
"""

from __future__ import annotations

import io
import json
import os
import re
import threading
from contextlib import contextmanager
from pathlib import Path
import urllib.parse
import urllib.request
from typing import Callable

import pandas as pd

from .data import DataError

BCE_URL = "https://data-api.ecb.europa.eu/service/data/EXR/D.{ccy}.EUR.SP00.A"
EODHD_URL = "https://eodhd.com/api/eod/{symbol}"
# Yahoo suffix -> EODHD exchange code (a ticker without suffix is a US listing)
EODHD_EXCHANGES = {"PA": "PA", "DE": "XETRA", "AS": "AS", "MI": "MI", "L": "LSE", "SW": "SW", "BR": "BR",
                   "MC": "MC", "LS": "LS", "F": "F", "VI": "VI", "ST": "ST", "CO": "CO", "HE": "HE", "OL": "OL",
                   "TO": "TO"}


def source_prix() -> str:
    return (os.environ.get("PA_SOURCE_PRIX") or "yahoo").strip().lower()


def source_change() -> str:
    chosen = (os.environ.get("PA_SOURCE_CHANGE") or "").strip().lower()
    return chosen or ("bce" if source_prix() != "yahoo" else "yahoo")


def libelle() -> str:
    """One line for the screens and the documents: who the figures come from."""
    prix = {"yahoo": "Yahoo Finance", "eodhd": "EODHD (données sous licence)"}.get(source_prix(), source_prix())
    change = {"yahoo": "Yahoo Finance", "bce": "taux de référence de la BCE (source : statistiques de la BCE)"
              }.get(source_change(), source_change())
    return f"Cours : {prix}. Taux de change : {change}."


def _get(url: str, params: dict | None = None, timeout: int = 30) -> bytes:
    if params:
        url += "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": "PortfolioAssistant/1.0", "Accept": "*/*"})
    with urllib.request.urlopen(request, timeout=timeout) as response:     # noqa: S310 (fixed https hosts)
        return response.read()


def _final(series: pd.Series) -> pd.Series:
    from .market_data import final_closes
    return final_closes(series)


# ----------------------------------------------------------------------------- exchange rates (ECB)
def is_pair(ticker: str) -> bool:
    return ticker.upper().endswith("=X") and len(ticker) == 8


def bce_download(ticker: str, start: str | None, end: str | None, *, get: Callable | None = None) -> pd.Series:
    """« EURUSD=X » -> dollars per euro, the ECB's daily reference rate (TARGET working days)."""
    ticker = ticker.upper()
    base, ccy = ticker[:3], ticker[3:6]
    if not is_pair(ticker) or base != "EUR":
        raise DataError(f"{ticker} : la BCE ne publie que des taux contre l'euro")
    params = {"format": "csvdata", "startPeriod": start or "1999-01-04"}
    if end:
        params["endPeriod"] = end
    try:
        raw = (get or _get)(BCE_URL.format(ccy=ccy), params)
    except Exception as exc:                                  # network, unknown currency
        raise DataError(f"taux {ticker} indisponible auprès de la BCE ({exc})") from exc
    frame = pd.read_csv(io.BytesIO(raw)) if raw.strip() else pd.DataFrame()
    if frame.empty or not {"TIME_PERIOD", "OBS_VALUE"} <= set(frame.columns):
        return pd.Series(dtype="float64", name=ticker)
    series = pd.Series(pd.to_numeric(frame["OBS_VALUE"], errors="coerce").values,
                       index=pd.to_datetime(frame["TIME_PERIOD"]), name=ticker).dropna()
    return _final(series)


# ----------------------------------------------------------------------------- prices (EODHD)
def eodhd_symbol(ticker: str) -> str:
    """Yahoo's spelling -> EODHD's: CW8.PA -> CW8.PA, SXR8.DE -> SXR8.XETRA, SPY -> SPY.US."""
    ticker = ticker.strip().upper()
    if "." not in ticker:
        return f"{ticker}.US"
    code, suffix = ticker.rsplit(".", 1)
    if suffix not in EODHD_EXCHANGES:
        raise DataError(f"{ticker} : place de cotation « .{suffix} » non prévue pour EODHD")
    return f"{code}.{EODHD_EXCHANGES[suffix]}"


def eodhd_download(ticker: str, start: str | None, end: str | None, *, get: Callable | None = None) -> pd.Series:
    """Adjusted closes (dividends and splits folded in) from EODHD."""
    key = os.environ.get("PA_EODHD_CLE", "").strip()
    if not key:
        raise DataError("PA_SOURCE_PRIX=eodhd demande la clé PA_EODHD_CLE (réglage du serveur)")
    params = {"api_token": key, "fmt": "json", "period": "d"}
    if start:
        params["from"] = start
    if end:
        params["to"] = end
    try:
        rows = json.loads((get or _get)(EODHD_URL.format(symbol=eodhd_symbol(ticker)), params))
    except DataError:
        raise
    except Exception as exc:
        raise DataError(f"cours {ticker} indisponible auprès d'EODHD ({type(exc).__name__})") from exc
    if not rows:
        return pd.Series(dtype="float64", name=ticker)
    frame = pd.DataFrame(rows)
    series = pd.Series(pd.to_numeric(frame["adjusted_close"], errors="coerce").values,
                       index=pd.to_datetime(frame["date"]), name=ticker).dropna()
    return _final(series)


# ----------------------------------------------------------------------------- prices brought by the cabinet
# A cabinet that already holds licensed prices (Quantalys, Morningstar, its custodian) or follows
# funds that no public source quotes can import them: date, code, price. For that cabinet only,
# an imported code is read from its own file, before any download, and never enters the shared
# cache. The folder is set for the current page run (one thread per run).
_local = threading.local()
CODE = re.compile(r"^[A-Z0-9][A-Z0-9.\-=^]{0,19}$")


def utiliser_cours(folder: str | Path | None) -> None:
    _local.folder = Path(folder) if folder else None


@contextmanager
def sans_cours_importes():
    """For what is shared between cabinets (the universes' price panels): downloads only."""
    previous = getattr(_local, "folder", None)
    _local.folder = None
    try:
        yield
    finally:
        _local.folder = previous


def cours_importe(ticker: str) -> pd.Series | None:
    folder = getattr(_local, "folder", None)
    ticker = ticker.strip().upper()
    if folder is None or not CODE.match(ticker):
        return None
    path = folder / f"{ticker}.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    return _final(frame.iloc[:, 0].astype("float64").rename(ticker)) if not frame.empty else None


def lire_import(data: bytes, filename: str) -> pd.DataFrame:
    """A CSV or Excel file with a date, a code (Yahoo code or ISIN) and a price per row, whatever
    the names and order of the columns -> clean rows (date, code, cours). French and English
    numbers and dates are read; a row that cannot be read is refused with its line number."""
    from .statements import parse_number
    name = filename.lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            raw = pd.read_excel(io.BytesIO(data), dtype=str)
        else:
            text = data.decode("utf-8-sig", errors="replace")
            sep = ";" if text.count(";") >= text.count(",") and ";" in text else ("\t" if "\t" in text else ",")
            raw = pd.read_csv(io.StringIO(text), sep=sep, dtype=str)
    except Exception as exc:
        raise DataError(f"fichier illisible ({exc})") from exc
    cols = {str(c).strip().lower(): c for c in raw.columns}

    def pick(*words):
        return next((orig for low, orig in cols.items() if any(w in low for w in words)), None)
    cdate, ccode = pick("date"), pick("code", "isin", "ticker", "symbole")
    cprice = pick("cours", "prix", "vl", "valeur liquidative", "clôture", "cloture", "close", "price", "nav")
    if not (cdate and ccode and cprice):
        raise DataError("il faut trois colonnes : une date, un code (code Yahoo ou ISIN) et un cours")
    rows = []
    for n, r in enumerate(raw.to_dict("records"), start=2):
        if not str(r[ccode] or "").strip() or str(r[ccode]) == "nan":
            continue
        code = str(r[ccode]).strip().upper()
        try:
            text = str(r[cdate]).strip()
            day = pd.to_datetime(text, dayfirst=not re.match(r"^\d{4}-", text))
            price = parse_number(str(r[cprice]))
        except Exception:
            raise DataError(f"ligne {n} : date ou cours illisible") from None
        if not CODE.match(code) or price is None or price <= 0 or pd.isna(day):
            raise DataError(f"ligne {n} : code, date ou cours invalide")
        rows.append({"date": day.normalize(), "code": code, "cours": float(price)})
    if not rows:
        raise DataError("aucune ligne de cours dans ce fichier")
    return pd.DataFrame(rows)


def enregistrer_cours(folder: str | Path, rows: pd.DataFrame) -> dict[str, int]:
    """Adds the rows to the cabinet's own price files (a date imported again replaces the old
    price); returns {code: number of days held}."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    for code, part in rows.groupby("code"):
        new = part.set_index("date")["cours"]
        path = folder / f"{code}.csv"
        if path.exists():
            old = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
            new = pd.concat([old[~old.index.isin(new.index)], new])
        new = new[~new.index.duplicated(keep="last")].sort_index().rename(code)
        new.to_frame().to_csv(path)
        out[code] = len(new)
    return out


def cours_du_cabinet(folder: str | Path) -> pd.DataFrame:
    """What the cabinet has imported: one row per code (days, first and last date, last price)."""
    folder = Path(folder)
    rows = []
    for path in sorted(folder.glob("*.csv")) if folder.exists() else []:
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        if len(s):
            rows.append({"code": path.stem, "jours": len(s), "du": s.index[0].date(), "au": s.index[-1].date(),
                         "dernier cours": float(s.iloc[-1])})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- the one entry point
def download(ticker: str, start: str | None, end: str | None) -> pd.Series:
    """The downloader the whole software uses: picks the source from the settings above."""
    from .market_data import _yahoo_download
    if is_pair(ticker):
        if source_change() == "bce" and ticker.upper().startswith("EUR"):
            return bce_download(ticker, start, end)
        return _yahoo_download(ticker, start, end)
    if source_prix() == "eodhd":
        return eodhd_download(ticker, start, end)
    if source_prix() != "yahoo":
        raise DataError(f"PA_SOURCE_PRIX={source_prix()} : source inconnue (yahoo ou eodhd)")
    return _yahoo_download(ticker, start, end)
