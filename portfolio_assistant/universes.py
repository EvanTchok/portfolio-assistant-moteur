"""Investment universes by wrapper: what a French adviser can actually put in each account.

* ``pea``   : Plan d'Épargne en Actions. Only EU-eligible equities: the world, the US and
              the Nasdaq come through SYNTHETIC ETFs (a swap delivers the index return).
              No bonds, no gold, no money-market fund: the only refuge is uninvested cash.
* ``monde`` : compte-titres / assurance-vie, no restriction: equities of every region,
              euro government bonds, gold, real estate, commodities. The US-listed ETFs
              are used for their LONG history (2006 onwards) and converted to euros; a
              French retail client cannot buy them (no PRIIPs KID) and would hold their
              UCITS equivalents, which track the same indices.
* ``ucits`` : the same asset classes as ``monde`` through funds a French retail client can
              buy (euro listings on Xetra). Shorter history (May 2010), and its test period
              counts as already consulted through ``monde`` (same indices).
* ``demo``  : the 5 US ETFs in dollars used by the first demos (kept for continuity).

Every price is expressed in the investor's currency (EUR) before any calculation: a US
ETF bought by a French investor carries the dollar risk, and the backtest must too.

Inception dates and tickers were checked on justETF (October 2026). A universe starts
when its youngest asset starts; younger assets are listed in ``recent`` and left out of
the long-history universe.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .market_data import DEFAULT_CACHE, fetch_prices

UNIVERSES: dict[str, dict] = {
    "pea": {
        "libellé": "PEA (actions éligibles uniquement)",
        "devise": "EUR",
        "journal": "laboratoire/research_pea",
        "actifs": [
            {"ticker": "CW8.PA", "nom": "Amundi MSCI World Swap", "classe": "actions monde",
             "devise": "EUR", "réplication": "synthétique", "depuis": "2009-06-16"},
            {"ticker": "ESE.PA", "nom": "BNP Paribas Easy S&P 500", "classe": "actions USA",
             "devise": "EUR", "réplication": "synthétique", "depuis": "2013-09-16"},
            {"ticker": "PUST.PA", "nom": "Amundi PEA Nasdaq-100", "classe": "actions USA",
             "devise": "EUR", "réplication": "synthétique", "depuis": "2014-05-20"},
            {"ticker": "ETZ.PA", "nom": "BNP Paribas Easy STOXX Europe 600", "classe": "actions Europe",
             "devise": "EUR", "réplication": "synthétique", "depuis": "2013-09-16"},
            {"ticker": "CAC.PA", "nom": "Amundi CAC 40", "classe": "actions Europe",
             "devise": "EUR", "réplication": "physique", "depuis": "2000-12-13"},
        ],
        "recent": [
            {"ticker": "PAEEM.PA", "nom": "Amundi PEA Emergent", "classe": "actions émergents",
             "devise": "EUR", "réplication": "synthétique", "depuis": "2019-04-25"},
            {"ticker": "PTPXE.PA", "nom": "Amundi PEA Japon (TOPIX)", "classe": "actions Japon",
             "devise": "EUR", "réplication": "synthétique", "depuis": "2019-04-25"},
        ],
        "remarques": [
            "Pas d'obligations, pas d'or, pas de fonds monétaire : le seul refuge est le cash, qui ne rapporte rien.",
            "Les ETF monde, S&P 500 et Nasdaq du PEA sont synthétiques. Le gouvernement a renoncé le 26/08/2026 "
            "à les exclure du PEA ; un amendement reste possible pendant l'examen du budget 2027 : à surveiller.",
            "Émergents et Japon n'existent en PEA que depuis 2019 : trop récents pour un historique long.",
        ],
    },
    "monde": {
        "libellé": "Compte-titres / assurance-vie (monde entier, toutes classes d'actifs)",
        "devise": "EUR",
        "journal": "laboratoire/research_monde",
        "actifs": [
            {"ticker": "SPY", "nom": "S&P 500 (SPDR)", "classe": "actions", "devise": "USD", "depuis": "1993-01-22"},
            {"ticker": "EXSA.DE", "nom": "iShares STOXX Europe 600", "classe": "actions", "devise": "EUR",
             "depuis": "2004-02-13"},
            {"ticker": "EWJ", "nom": "MSCI Japan (iShares)", "classe": "actions", "devise": "USD", "depuis": "1996-03-12"},
            {"ticker": "EEM", "nom": "MSCI Emerging Markets (iShares)", "classe": "actions", "devise": "USD",
             "depuis": "2003-04-07"},
            {"ticker": "EXHA.DE", "nom": "iShares eb.rexx Government Germany (1,5-10,5 ans)",
             "classe": "obligations", "devise": "EUR", "depuis": "2003-02-04"},
            {"ticker": "GLD", "nom": "Or (SPDR Gold Shares)", "classe": "or", "devise": "USD", "depuis": "2004-11-18"},
            {"ticker": "VNQ", "nom": "Immobilier coté US (Vanguard REIT)", "classe": "immobilier", "devise": "USD",
             "depuis": "2004-09-23"},
            {"ticker": "DBC", "nom": "Matières premières (Invesco DB Commodity)", "classe": "matières premières",
             "devise": "USD", "depuis": "2006-02-03"},
        ],
        "recent": [],
        "remarques": [
            "Les ETF cotés aux États-Unis servent d'historique long (depuis 2006). Un client français ne peut pas "
            "les acheter (pas de DIC PRIIPs) : il détiendrait leurs équivalents UCITS sur les mêmes indices.",
            "Tous les prix sont convertis en euros : le risque de change dollar fait partie du résultat.",
            "Le refuge est l'obligataire d'État allemand (EXHA.DE).",
        ],
    },
    "ucits": {
        "libellé": "Fonds UCITS investissables (compte-titres / assurance-vie)",
        "devise": "EUR",
        "journal": "laboratoire/research_ucits",
        # same indices as « monde »: its look at 2023 onwards counts here too
        "test_déjà_vu_via": ["monde"],
        "actifs": [
            {"ticker": "SXR8.DE", "nom": "iShares Core S&P 500", "classe": "actions", "devise": "EUR",
             "isin": "IE00B5BMR087", "depuis": "2010-05-19"},
            {"ticker": "EXSA.DE", "nom": "iShares STOXX Europe 600", "classe": "actions", "devise": "EUR",
             "depuis": "2004-02-13"},
            {"ticker": "SXR5.DE", "nom": "iShares MSCI Japan", "classe": "actions", "devise": "EUR",
             "isin": "IE00B53QDK08", "depuis": "2010-01-11"},
            {"ticker": "EUNM.DE", "nom": "iShares MSCI EM", "classe": "actions", "devise": "EUR",
             "isin": "IE00B4L5YC18", "depuis": "2009-09-25"},
            {"ticker": "EXHA.DE", "nom": "iShares eb.rexx Government Germany (1,5-10,5 ans)",
             "classe": "obligations", "devise": "EUR", "depuis": "2003-02-04"},
            {"ticker": "4GLD.DE", "nom": "Xetra-Gold (ETC adossé à de l'or physique)", "classe": "or",
             "devise": "EUR", "isin": "DE000A0S9GB0", "depuis": "2007-11-27"},
            {"ticker": "IQQ6.DE", "nom": "iShares Developed Markets Property Yield", "classe": "immobilier",
             "devise": "EUR", "isin": "IE00B1FZS350", "depuis": "2006-10-20"},
            {"ticker": "EXXY.DE", "nom": "iShares Diversified Commodity Swap", "classe": "matières premières",
             "devise": "EUR", "isin": "DE000A0H0728", "depuis": "2007-08-07"},
        ],
        "recent": [],
        "remarques": [
            "Les mêmes classes d'actifs que l'univers Monde, mais avec des fonds qu'un particulier français "
            "peut acheter : cotés en euros sur Xetra, avec un DIC. L'historique commun commence en mai 2010 "
            "(S&P 500 UCITS) : plus court que Monde.",
            "Xetra-Gold est un ETC (titre de créance adossé à de l'or physique), pas un fonds UCITS ; il est "
            "accessible à un particulier en compte-titres.",
            "Ces fonds suivent les mêmes indices que l'univers Monde, dont la période 2023 → aujourd'hui a "
            "déjà été consultée : un test ici est donc une confirmation affaiblie, jamais un test propre.",
        ],
    },
    "demo": {
        "libellé": "Démo : 5 ETF américains en dollars",
        "devise": "USD",
        "journal": "laboratoire/research_demo",
        "actifs": [
            {"ticker": "SPY", "classe": "actions", "devise": "USD"},
            {"ticker": "EFA", "classe": "actions", "devise": "USD"},
            {"ticker": "EEM", "classe": "actions", "devise": "USD"},
            {"ticker": "AGG", "classe": "obligations", "devise": "USD"},
            {"ticker": "GLD", "classe": "or", "devise": "USD"},
        ],
        "recent": [],
        "remarques": ["Univers des premières démos, en dollars, sans conversion."],
    },
}


# ----------------------------------------------------------- custom universes
# An adviser brings his own funds: each custom universe is a JSON file in a folder of the
# project (univers/ by default), with its own journal. Once its journal holds a trial, its
# assets can no longer change: the trials would no longer describe the same universe.
CUSTOM_DIR = Path("donnees_cabinet") / "univers"
CLASSES = ["actions", "obligations", "or", "immobilier", "matières premières", "monétaire", "autre"]
CURRENCIES = ["EUR", "USD", "GBP", "CHF", "JPY"]


def use_folder(path: str | Path) -> None:
    """Where custom universes are read and written (the project's univers/ folder)."""
    global CUSTOM_DIR
    CUSTOM_DIR = Path(path)


def _customs() -> dict[str, dict]:
    import json
    out = {}
    if CUSTOM_DIR.exists():
        for path in sorted(CUSTOM_DIR.glob("*.json")):
            try:
                u = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue                      # a hand-broken file must not break the app
            if path.stem not in UNIVERSES:
                out[path.stem] = {**u, "personnalisé": True}
    return out


def names() -> list[str]:
    return list(UNIVERSES) + list(_customs())


def get(name: str) -> dict:
    if name in UNIVERSES:
        return UNIVERSES[name]
    customs = _customs()
    if name in customs:
        return customs[name]
    raise ValueError(f"univers inconnu « {name} » ; univers disponibles : {', '.join(names())}")


def slug(text: str) -> str:
    import re
    import unicodedata
    ascii_ = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_.lower()).strip("_")[:40]


def validate_custom(key: str, label: str, items: list[dict], same_indices_as: list[str] | None = None) -> list[str]:
    """What is wrong with a custom universe, in plain French (empty list: it is valid)."""
    errors = []
    if not key:
        errors.append("donnez un nom court à l'univers (lettres et chiffres)")
    elif key in UNIVERSES:
        errors.append(f"« {key} » est le nom d'un univers fourni : choisissez-en un autre")
    if not str(label).strip():
        errors.append("donnez un libellé à l'univers (ce que verra le conseiller)")
    tickers = [str(a.get("ticker", "")).strip().upper() for a in items]
    if len([t for t in tickers if t]) < 2:
        errors.append("il faut au moins 2 actifs")
    dupes = sorted({t for t in tickers if t and tickers.count(t) > 1})
    if dupes:
        errors.append("codes en double : " + ", ".join(dupes))
    for a in items:
        if not str(a.get("ticker", "")).strip():
            continue
        if a.get("classe") not in CLASSES:
            errors.append(f"{a['ticker']} : classe « {a.get('classe')} » inconnue ({', '.join(CLASSES)})")
        if a.get("devise") not in CURRENCIES:
            errors.append(f"{a['ticker']} : devise « {a.get('devise')} » non gérée ({', '.join(CURRENCIES)})")
    if not any(a.get("classe") == "actions" for a in items if str(a.get("ticker", "")).strip()):
        errors.append("il faut au moins un actif de classe « actions » : la référence des profils en a besoin")
    for other in same_indices_as or []:
        if other not in names():
            errors.append(f"univers « {other} » inconnu")
    return errors


def is_locked(name: str, root: str | Path = ".") -> bool:
    """True once the universe's journal holds anything: its assets are then fixed."""
    log = journal(name, root) / "registry.jsonl"
    return log.exists() and log.stat().st_size > 0


def save_custom(key: str, label: str, items: list[dict], *, same_indices_as: list[str] | None = None,
                remarks: list[str] | None = None, root: str | Path = ".") -> Path:
    """Writes univers/<key>.json. Refused if invalid, or if the universe already has trials."""
    import json
    errors = validate_custom(key, label, items, same_indices_as)
    if errors:
        raise ValueError("univers invalide :\n  - " + "\n  - ".join(errors))
    if key in _customs() and is_locked(key, root):
        raise ValueError(f"l'univers « {key} » a déjà des essais à son registre : ses actifs ne peuvent plus "
                         "changer. Créez un nouvel univers sous un autre nom.")
    clean = [{"ticker": str(a["ticker"]).strip().upper(), "nom": str(a.get("nom") or "").strip(),
              "classe": a["classe"], "devise": a["devise"]}
             for a in items if str(a.get("ticker", "")).strip()]
    data = {"libellé": str(label).strip(), "devise": "EUR", "journal": f"laboratoire/research_{key}", "actifs": clean,
            "recent": [], "remarques": list(remarks or [])}
    if same_indices_as:
        data["test_déjà_vu_via"] = list(same_indices_as)
        data["remarques"].append("Mêmes indices que " + ", ".join(get(o)["libellé"] for o in same_indices_as)
                                 + " : la période de test compte comme déjà consultée.")
    CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
    path = CUSTOM_DIR / f"{key}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def check_prices(items: list[dict], start: str = "2005-01-01", *, cache_dir: str | Path = DEFAULT_CACHE,
                 fetcher=None) -> pd.DataFrame:
    """Each asset's price history as found (first and last date, number of prices), before
    anything is saved: a wrong code shows up here, not in the middle of a backtest."""
    from .market_data import fetch_ticker
    fetch = fetcher or (lambda t: fetch_ticker(t, start, cache_dir=cache_dir))
    rows = []
    for a in items:
        t = str(a.get("ticker", "")).strip().upper()
        if not t:
            continue
        try:
            s = fetch(t).dropna()
            rows.append({"ticker": t, "trouvé": "oui", "premier prix": s.index[0].date(),
                         "dernier prix": s.index[-1].date(), "nombre de prix": len(s)})
        except Exception as exc:                # any failure is reported, nothing is saved
            reason = "aucun prix : vérifiez le code et sa place de cotation" if "no price" in str(exc) \
                else str(exc)[:80]
            rows.append({"ticker": t, "trouvé": f"non ({reason})", "premier prix": None,
                         "dernier prix": None, "nombre de prix": 0})
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()


def assets(name: str, *, include_recent: bool = False) -> list[dict]:
    """The universe as the strategy builder expects it (extra keys are ignored by it)."""
    u = get(name)
    return [dict(a) for a in u["actifs"] + (u["recent"] if include_recent else [])]


def table(name: str, *, include_recent: bool = True) -> pd.DataFrame:
    rows = assets(name, include_recent=include_recent)
    recent = {a["ticker"] for a in get(name)["recent"]}
    frame = pd.DataFrame(rows).set_index("ticker")
    frame["historique long"] = ["non (trop récent)" if t in recent else "oui" for t in frame.index]
    return frame


def journal(name: str, root: str | Path = ".") -> Path:
    return Path(root) / get(name)["journal"]


def prior_consultations(name: str, root: str | Path = ".") -> int:
    """Looks at the test period through universes on the same indices (their tests)."""
    import json
    count = 0
    for other in get(name).get("test_déjà_vu_via", []):
        log = journal(other, root) / "registry.jsonl"
        if log.exists():
            count += sum(1 for line in log.read_text(encoding="utf-8").splitlines()
                         if line.strip() and json.loads(line).get("event") == "oos")
    return count


def load_prices(name: str, start: str = "2005-01-01", *, include_recent: bool = False,
                cache_dir: str | Path = DEFAULT_CACHE, verbose: bool = True, **kwargs) -> pd.DataFrame:
    """Prices of the universe, every column in the universe's currency."""
    u = get(name)
    items = assets(name, include_recent=include_recent)
    currencies = {a["ticker"]: a.get("devise", u["devise"]) for a in items}
    return fetch_prices([a["ticker"] for a in items], start=start, cache_dir=cache_dir,
                        verbose=verbose, currencies=currencies, base_currency=u["devise"], **kwargs)
