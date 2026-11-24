"""Strategies written as Python code, and an AI assistant that drafts them.

A rule is a file ``laboratoire/code/<name>.py`` that defines ``decide(ctx)``:

    def decide(ctx):
        h = ctx.history                      # prices up to the decision date, never after
        ...
        return {"CW8.PA": 0.6, "ESE.PA": 0.4}

It becomes the lab rule ``code:<name>``, explored, frozen, tested and tracked like the
others. Its fingerprint is the hash of the file's text: editing a frozen rule's file is
seen as a changed code by the test and by the tracking, as for the built-in rules.

Writing or running NEW code is only offered on a single computer (never online): Python
cannot be sandboxed, so a server must not execute code typed into a web page. The files
already in the project are part of the software, like any other module.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import math
import os
import re
import statistics
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

PREFIX = "code:"
ALLOWED_MODULES = {"numpy": np, "pandas": pd, "math": math, "statistics": statistics}
TEMPLATE = '''"""Ma stratégie : moitié sur les 2 lignes les plus régulières des 6 derniers mois."""


def decide(ctx):
    h = ctx.history                          # cours jusqu'à la date de décision, jamais après
    if len(h) < 130:
        return None                          # pas encore assez d'historique : on reste en liquidités
    r = h.pct_change().iloc[-126:]
    score = r.mean() / r.std()               # rendement par unité de risque sur 6 mois
    best = score.sort_values().index[-2:]
    return {t: 0.5 for t in best}            # poids entre 0 et 1, somme au plus 1
'''
EXAMPLES = {
    "Les 2 plus réguliers sur 6 mois": TEMPLATE,
    "Tendance avec refuge": '''"""Actions au-dessus de leur moyenne 200 jours à parts égales ; le reste en liquidités."""


def decide(ctx):
    h = ctx.history
    if len(h) < 201:
        return None
    above = h.iloc[-1] > h.iloc[-200:].mean()
    chosen = above[above].index
    if len(chosen) == 0:
        return {}
    return {t: 1 / len(h.columns) for t in chosen}
''',
    "Volatilité cible 10 %": '''"""Poids égaux, exposition réduite quand la volatilité du panier dépasse 10 % par an."""
import numpy as np


def decide(ctx):
    h = ctx.history
    if len(h) < 64:
        return None
    r = h.pct_change().iloc[-63:].mean(axis=1)
    vol = r.std() * np.sqrt(252)
    exposure = min(1.0, 0.10 / vol) if vol > 0 else 1.0
    return {t: exposure / len(h.columns) for t in h.columns}
''',
}


class CodeError(ValueError):
    """The code cannot become a rule (syntax, missing decide, forbidden import)."""


def editor_allowed() -> bool:
    """Never on a server open to others (PA_EN_LIGNE=1): Python cannot be sandboxed."""
    return os.environ.get("PA_EN_LIGNE", "").strip().lower() not in ("1", "oui", "true", "yes")


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(name).strip().lower()
               .translate(str.maketrans("àâäéèêëîïôöùûüç", "aaaeeeeiioouuuc"))).strip("-")
    if not s:
        raise CodeError("donnez un nom à la stratégie")
    return s[:60]


def fingerprint(source: str) -> str:
    return hashlib.sha256(source.replace("\r\n", "\n").strip().encode()).hexdigest()[:16]


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in ALLOWED_MODULES:
        raise ImportError(f"module « {name} » non autorisé ici (autorisés : {', '.join(ALLOWED_MODULES)})")
    return __import__(name, globals, locals, fromlist, level)


_SAFE = {k: getattr(builtins, k) for k in (
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "isinstance", "len", "list",
    "map", "max", "min", "pow", "range", "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip",
    "ValueError", "KeyError", "Exception", "None", "True", "False", "print")} | {"__import__": _import}


def compile_rule(source: str):
    """``decide`` from the source; refuses what cannot run. (A guard against mistakes, not
    a sandbox: see the module's note.)"""
    try:
        code = compile(source, "<stratégie>", "exec")
    except SyntaxError as exc:
        raise CodeError(f"erreur de syntaxe ligne {exc.lineno} : {exc.msg}") from None
    namespace = {"__builtins__": _SAFE, "np": np, "pd": pd, "math": math, "__name__": "strategie"}
    try:
        exec(code, namespace)
    except ImportError as exc:
        raise CodeError(str(exc)) from None
    except Exception as exc:
        raise CodeError(f"le code s'arrête au chargement : {type(exc).__name__} : {exc}") from None
    decide = namespace.get("decide")
    if not callable(decide):
        raise CodeError("le code doit définir une fonction decide(ctx)")
    return decide


def factory_for(source: str):
    decide = compile_rule(source)

    def factory(**params):
        return decide
    factory.__code_fingerprint__ = fingerprint(source)
    return factory


# ------------------------------------------------------------------------------ files
def folder(lab_dir: str | Path) -> Path:
    return Path(lab_dir) / "code"


def save(lab_dir: str | Path, name: str, source: str) -> str:
    compile_rule(source)
    f = folder(lab_dir)
    f.mkdir(parents=True, exist_ok=True)
    key = slug(name)
    (f / f"{key}.py").write_text(source.replace("\r\n", "\n"), encoding="utf-8")
    return PREFIX + key


def saved(lab_dir: str | Path) -> dict[str, str]:
    f = folder(lab_dir)
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(f.glob("*.py"))} if f.exists() else {}


def load_all(lab_dir: str | Path) -> dict:
    """Every rule file as a lab rule; a broken file is skipped (the lab says the rule is unknown)."""
    out = {}
    for key, source in saved(lab_dir).items():
        try:
            out[PREFIX + key] = factory_for(source)
        except CodeError:
            continue
    return out


# -------------------------------------------------------------------------- assistant
ASSISTANT_SYSTEM = """Tu écris des règles d'allocation pour un laboratoire de backtest français.
Réponds UNIQUEMENT par un bloc de code Python qui définit decide(ctx), sans autre texte.
ctx.history : DataFrame pandas des cours de clôture (colonnes = tickers, index = dates) jusqu'à
la date de décision incluse ; ctx.current_weights : Series des poids détenus ; ctx.nav : valeur.
decide renvoie None (rester en place au début), ou un dict {ticker: poids}, poids >= 0, somme <= 1
(le reste en liquidités). N'utilise que numpy (np), pandas (pd), math et statistics. N'accède à
aucun fichier ni réseau. Commente en français, simplement. Pas de look-ahead : n'utilise que h."""


def assistant_ready() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY")) and editor_allowed()


def ask_assistant(request: str, tickers: list[str], current: str = "", *, timeout: float = 60.0) -> str:
    """Drafts a rule from a description, with the user's own API key (ANTHROPIC_API_KEY,
    which the user sets on his computer). The draft is shown, never run on its own."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise CodeError("assistant indisponible : la variable ANTHROPIC_API_KEY n'est pas définie sur cet ordinateur")
    prompt = (f"Tickers disponibles : {', '.join(tickers)}.\n"
              + (f"Code actuel :\n```python\n{current}\n```\n" if current.strip() else "")
              + f"Demande : {request}")
    body = json.dumps({"model": os.environ.get("PA_IA_MODELE", "claude-sonnet-5-5"), "max_tokens": 2000,
                       "system": ASSISTANT_SYSTEM, "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, method="POST",
                                 headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                                          "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:
        raise CodeError(f"l'assistant n'a pas répondu ({type(exc).__name__})") from None
    text = "".join(part.get("text", "") for part in data.get("content", []) if part.get("type") == "text")
    return extract_code(text)


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip() + "\n"
