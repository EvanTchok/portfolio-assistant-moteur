"""The research report: everything the lab recorded, in one PDF for a professional reader.

    from portfolio_assistant.synthesis import build_report
    build_report(project_root, "rapports/rapport_de_recherche.pdf")

Nothing in it is typed by hand: every figure comes from the journals the lab and the
tracker wrote (research_*/registry.jsonl, suivi/), the rules come from the code that
applied them (profiles, materiality threshold), and the chronology comes from git. Run
it again after a new test or a new tracking point and the report follows.

Test-period curves are recomputed from the current prices (they are illustrations); the
verdicts, gaps and statuses shown are the ones recorded in the journal at test time.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Callable

import pandas as pd

from . import profiles as _profiles
from . import chemins
from . import universes as _universes
from .compare import MATERIALITY
from .metrics import summary

OOS_START = "2023-01-01"

# the pass marks of scripts/finalists.py, restated for the reader
ROBUSTNESS_RULES = [
    ("Réglages", "part des réglages voisins qui battent la référence", "≥ 70 %"),
    ("Frais", "frais par transaction qui effacent l'avantage", "≥ 0,5 %"),
    ("Fenêtres", "part des périodes de 3 ans gagnées contre la référence", "≥ 60 %"),
    ("Bootstrap", "probabilité de battre la référence sur 2 000 histoires rejouées", "≥ 60 %"),
    ("Chance", "Sharpe dégonflé (Bailey & López de Prado), tous essais comptés", "≥ 90 %"),
]

STATUS_TEXT = {
    "VALID_OOS": "test propre : gelée avant que quiconque regarde la période de test",
    "HOLDOUT_ALREADY_CONSULTED": "confirmation affaiblie : la période de test avait déjà été consultée",
    "REPEAT": "test relancé : seul le premier compte",
    "CONTAMINATED_CODE_CHANGED": "invalide : le code a changé après le gel",
}
VERDICT_TEXT = {
    "ADDS_VALUE": "apporte de la valeur",
    "ADDS_NOTHING": "n'apporte rien",
    "NO_REAL_DIFFERENCE": "égalité",
    "UNSUITABLE": "inadaptée au profil",
    "INSUFFICIENT_SAMPLE": "échantillon insuffisant",
}

SHORT_VERDICT = {"ADDS_VALUE": "valeur", "ADDS_NOTHING": "rien", "NO_REAL_DIFFERENCE": "égalité",
                 "UNSUITABLE": "inadaptée", "INSUFFICIENT_SAMPLE": "insuffisant"}


# ===================================================================== data collection
def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _profile_label(name: str | None) -> str:
    return _profiles.get(name).label if name else "sans profil (1/N, γ=3)"


def collect(root: str | Path) -> dict:
    """Everything the report needs, read from the project folder."""
    root = Path(root)
    journals = {}
    for name, u in ((n, _universes.get(n)) for n in _universes.names()):
        events = _read_jsonl(root / u["journal"] / "registry.jsonl")
        if not events:
            continue
        trials = {}
        for e in events:
            if e["event"] == "trial":
                trials.setdefault((e["fingerprint"], e.get("profile")), e)
        freezes = [e for e in events if e["event"] == "freeze"]
        tests = []
        for t in (e for e in events if e["event"] == "oos"):
            frozen = [f for f in freezes if f["name"] == t["name"] and f.get("profile") == t.get("profile")
                      and f["fingerprint"] == t["fingerprint"]]
            tests.append({"universe": name, "profile": t.get("profile"), "name": t["name"],
                          "status": t["status"], "tested_at": t["at"], "oos": t["oos"],
                          "freeze": frozen[-1] if frozen else None})
        journals[name] = {"label": u["libellé"], "trials": list(trials.values()),
                          "freezes": freezes, "tests": tests}

    tracker_dir = root / "laboratoire" / "suivi"
    tracked = []
    if tracker_dir.exists():
        history = _read_jsonl(tracker_dir / "journal.jsonl")
        for path in sorted(tracker_dir.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            points = [h for h in history if h.get("event") == "update" and h.get("nom") == record["nom"]]
            tracked.append({"record": record, "points": points})
    return {"journals": journals, "tracked": tracked, "git": git_log(root),
            "generated": dt.datetime.now()}


def git_log(root: str | Path) -> list[dict]:
    """The project's commits, oldest first (empty if git is not available)."""
    try:
        out = subprocess.run(["git", "-C", str(root), "log", "--reverse", "--date=format:%Y-%m-%d %H:%M",
                              "--format=%h|%ad|%s"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    rows = []
    for line in out.stdout.splitlines():
        parts = line.split("|", 2)
        if len(parts) == 3:
            rows.append({"commit": parts[0], "date": parts[1], "message": parts[2]})
    return rows


def git_author(root: str | Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(root), "config", "user.name"], capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def test_curves(test: dict, root: Path, price_loader: Callable, cache_dir) -> dict | None:
    """Strategy and reference NAV over the test period, recomputed (not logged anywhere)."""
    from .research import Lab, StrategySpec

    if not test["freeze"]:
        return None
    prices = price_loader(test["universe"], cache_dir=cache_dir, verbose=False)
    from . import chemins, code_rules
    lab = Lab(prices, OOS_START, path=root / _universes.get(test["universe"])["journal"],
              profile=test["profile"], rules=code_rules.load_all(chemins.lab(root)))
    spec = StrategySpec.from_dict(test["freeze"]["spec"])
    out = lab._run(spec, lab.prices, start=lab.oos_start)
    end = pd.Timestamp(test["tested_at"][:10])
    strategy, baseline = out["results"]
    navs = pd.DataFrame({spec.name: strategy.nav, out["reference"]: baseline.nav}).loc[out["start"]:end]
    return {"navs": navs / navs.iloc[0] * 100, "reference": out["reference"],
            "spec": spec, "summary": {c: summary(navs[c]) for c in navs}}


# ========================================================================= charts
def _chart(curves: dict, path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    navs = curves["navs"]
    dd = navs / navs.cummax() - 1
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.2, 4.2), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1.4]})
    colors = ["#1F3864", "#C0504D"]
    for (name, s), c in zip(navs.items(), colors):
        top.plot(s.index, s.values, color=c, lw=1.6, label=name)
        bottom.plot(dd.index, dd[name].values * 100, color=c, lw=1.1)
    top.set_ylabel("base 100")
    top.legend(frameon=False, fontsize=8, loc="upper left")
    bottom.set_ylabel("chute (%)")
    for ax in (top, bottom):
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


# =========================================================================== PDF
def _fonts():
    import matplotlib
    from reportlab.lib.fonts import addMapping
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    base = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    for name, file in (("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"),
                       ("DV-I", "DejaVuSans-Oblique.ttf"), ("DV-BI", "DejaVuSans-BoldOblique.ttf")):
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(base / file)))
    addMapping("DV", 0, 0, "DV"), addMapping("DV", 1, 0, "DV-B")
    addMapping("DV", 0, 1, "DV-I"), addMapping("DV", 1, 1, "DV-BI")


NAVY, LIGHT, GREY = "#1F3864", "#E9EEF6", "#6B7280"


def _styles():
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle

    s = {}
    s["title"] = ParagraphStyle("title", fontName="DV-B", fontSize=26, leading=31, textColor=colors.HexColor(NAVY))
    s["subtitle"] = ParagraphStyle("subtitle", fontName="DV", fontSize=13, leading=18, textColor=colors.HexColor(GREY))
    s["h1"] = ParagraphStyle("h1", fontName="DV-B", fontSize=16, leading=21, spaceBefore=6, spaceAfter=8,
                             textColor=colors.HexColor(NAVY))
    s["h2"] = ParagraphStyle("h2", fontName="DV-B", fontSize=11.5, leading=15, spaceBefore=10, spaceAfter=4,
                             textColor=colors.HexColor(NAVY))
    s["body"] = ParagraphStyle("body", fontName="DV", fontSize=9.3, leading=13.2, spaceAfter=5)
    s["bullet"] = ParagraphStyle("bullet", parent=s["body"], leftIndent=12, bulletIndent=2, spaceAfter=2.5)
    s["small"] = ParagraphStyle("small", fontName="DV", fontSize=7.6, leading=10, textColor=colors.HexColor(GREY))
    s["cell"] = ParagraphStyle("cell", fontName="DV", fontSize=7.8, leading=9.8)
    s["cellb"] = ParagraphStyle("cellb", fontName="DV-B", fontSize=7.8, leading=9.8, textColor=colors.white)
    s["box"] = ParagraphStyle("box", fontName="DV", fontSize=9.3, leading=13.5)
    return s


def _table(rows, widths, st, *, header=True, zebra=True, align_right_from=None):
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, Table, TableStyle

    data = []
    for i, row in enumerate(rows):
        style = st["cellb"] if header and i == 0 else st["cell"]
        data.append([c if not isinstance(c, str) else Paragraph(c, style) for c in row])
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    cmds = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 3.5), ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
            ("LINEBELOW", (0, -1), (-1, -1), 0.4, colors.HexColor("#B8C2D6"))]
    if header:
        cmds.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)))
    if zebra:
        for r in range(1 if header else 0, len(rows)):
            if r % 2 == 0:
                cmds.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor(LIGHT)))
    t.setStyle(TableStyle(cmds))
    return t


def _pct(v, signed=False, digits=1):
    """French figures: 16,3 %, +1,9 %, −20,4 % (the reader is a French adviser)."""
    if v is None or (isinstance(v, float) and v != v):
        return "–"
    text = f"{v * 100:+.{digits}f}" if signed else f"{v * 100:.{digits}f}"
    return text.replace("-", "\u2212").replace(".", ",") + "\u202f%"


def _num(v, digits=2):
    if v is None or (isinstance(v, float) and v != v):
        return "–"
    return f"{v:.{digits}f}".replace("-", "\u2212").replace(".", ",")


_REBALANCE = {"monthly": "mensuel", "weekly": "hebdomadaire", "quarterly": "trimestriel",
              "daily": "quotidien", "yearly": "annuel", "annual": "annuel"}
_RULES = {"fixed_weights": "Allocation fixe", "equal_weight": "Poids égaux", "buy_and_hold": "Achat puis conservation",
          "inverse_volatility": "Inverse de la volatilité", "momentum_top_k": "Les k meilleures tendances"}


def _fr(text: str) -> str:
    """The builder's own description (kept unchanged: its code is fingerprinted) in French figures."""
    import re
    text = re.sub(r"(\d+)\.(\d+)\s?%", lambda m: f"{m.group(1)},{m.group(2)}\u202f%", text)
    text = re.sub(r"(\d+)%", lambda m: f"{m.group(1)}\u202f%", text)
    text = re.sub(r"= 0\.0\b", "= 0", text)
    text = re.sub(r"(\d+)\.(\d+)", r"\1,\2", text)
    for en, fr in _REBALANCE.items():
        text = re.sub(rf"\b{en}\b", fr, text)
    return text


def _fr_date(value) -> str:
    """2026-10-05 -> 05/10/2026 (anything unreadable is shown as it is)."""
    try:
        return pd.Timestamp(value).strftime("%d/%m/%Y")
    except (ValueError, TypeError):
        return str(value)


def track_title(label: str) -> str:
    """'Suivi de tendance · monde · equilibre' -> 'Suivi de tendance — Compte-titres / assurance-vie · Équilibré'."""
    from . import profiles as _profiles
    parts = label.split(" · ")
    name, rest = parts[0], parts[1:]
    shown = []
    for i, bit in enumerate(rest):
        if i == 0 and bit in _universes.names():
            shown.append(_universes.get(bit)["libellé"].split(" (")[0])
        elif bit in _profiles.PROFILES:
            shown.append(_profiles.get(bit).label)
        else:
            shown.append(bit)
    if not rest:
        shown.append(_universes.get("demo")["libellé"].split(" (")[0])
    return f"{name} — {' · '.join(shown)}"


_PART = {"selection": "sél.", "ponderation": "pond.", "contraintes": "contr.", "risque": "risque"}
_LEAF = {"days": "{} j", "lookback": "{} j", "k": "{} actif(s)", "max_weight": "max {:.0%}",
         "target_vol": "vol. cible {:.0%}", "threshold": "seuil {}", "max_vol": "vol. max {:.0%}"}


def readable_name(name: str) -> str:
    """'Markowitz {'ponderation.days': 126, 'contraintes.max_weight': 0.3}' ->
    'Markowitz (pond. 126 j, contr. max 30%)': grid variants, as a reader expects them."""
    import ast

    if "{" not in name:
        return name
    base, _, rest = name.partition(" {")
    try:
        settings = ast.literal_eval("{" + rest)
    except (ValueError, SyntaxError):
        return name
    parts = []
    for key, value in settings.items():
        bits = str(key).split(".")
        leaf = _LEAF.get(bits[-1], bits[-1] + " {}")
        try:
            text = leaf.format(value)
        except (ValueError, TypeError):
            text = f"{bits[-1]} {value}"
        prefix = _PART.get(bits[0], "")
        parts.append(f"{prefix} {text}".strip())
    return f"{base} ({', '.join(parts)})"


def _recipe(spec: dict) -> list[str]:
    """The strategy in plain French, for the report."""
    import re

    if spec["rule"] != "composite":
        def show(v):
            return ", ".join(f"{t} {w:.0%}" for t, w in v.items()) if isinstance(v, dict) else str(v)
        words = {"lookback": "période d'observation (jours)", "weights": "poids", "k": "nombre d'actifs"}
        params = ", ".join(f"{words.get(k, k)} = {show(v)}" for k, v in spec.get("params", {}).items()) or "aucun réglage"
        return [_fr(f"Règle : {_RULES.get(spec['rule'], spec['rule'])} ({params})"),
                f"Univers : {', '.join(spec.get('universe') or []) or 'tous les actifs'} ; "
                f"rebalancement : {_REBALANCE.get(spec.get('rebalance'), spec.get('rebalance'))}, "
                f"frais {_pct(spec.get('fee_rate', 0), digits=2)} par transaction"]
    from .builder import describe

    lines = describe(spec["params"]["definition"]).splitlines()[1:]
    if isinstance(spec.get("cash_rate"), str):
        lines.append(f"Liquidités : placées en fonds monétaire ({spec['cash_rate']}, €STR), au lieu de 0 %")
    out = []
    for line in lines:
        match = re.search(r"poids par actif = (\{.*\})", line)
        if match:
            import ast
            weights = ast.literal_eval(match.group(1))
            values = set(round(v, 6) for v in weights.values())
            text = (f"{next(iter(values)):.1%} sur chacun des {len(weights)} actifs" if len(values) == 1
                    else ", ".join(f"{t} {w:.0%}" for t, w in weights.items()))
            line = line[:match.start()] + "poids : " + text + line[match.end():]
        out.append(_fr(line))
    return out


def _esc(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _context(test: dict) -> str:
    return f"{_universes.get(test['universe'])['libellé'].split(' (')[0]} · {_profile_label(test['profile'])}"


def build_report(root: str | Path, path: str | Path, *, author: str | None = None,
                 price_loader: Callable | None = None, cache_dir=None) -> Path:
    """Writes the research report as a PDF and returns its path."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                    Spacer, Table, TableStyle)

    root, path = Path(root), Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    price_loader = price_loader or _universes.load_prices
    cache_dir = cache_dir or chemins.cache(root)
    author = author if author is not None else git_author(root)
    data = collect(root)
    _fonts()
    st = _styles()
    width = A4[0] - 36 * mm
    story = []
    P = lambda text, style="body": Paragraph(text, st[style])  # noqa: E731

    def bullets(items):
        for item in items:
            story.append(Paragraph(item, st["bullet"], bulletText="•"))

    journals = data["journals"]
    tests = [t for j in journals.values() for t in j["tests"]]
    n_trials = sum(len(j["trials"]) for j in journals.values())
    n_freezes = sum(len(j["freezes"]) for j in journals.values())
    n_valid = sum(t["status"] == "VALID_OOS" for t in tests)
    n_win = sum(t["oos"]["verdict"] == "ADDS_VALUE" for t in tests if t["status"] == "VALID_OOS")
    generated = data["generated"]

    # ------------------------------------------------------------------ cover
    story += [Spacer(1, 50 * mm), P("Portfolio Assistant", "title"), Spacer(1, 4 * mm),
              P("Rapport de recherche : stratégies d'allocation construites sur le passé, "
                "gelées, puis testées une seule fois sur des années jamais vues", "subtitle"),
              Spacer(1, 12 * mm)]
    facts = [["univers étudiés", str(len(journals))], ["configurations essayées et comptées", str(n_trials)],
             ["stratégies gelées", str(n_freezes)], ["tests uniques hors échantillon", str(len(tests))],
             ["dont tests propres (VALID_OOS)", str(n_valid)], ["stratégies suivies en temps réel", str(len(data["tracked"]))]]
    cover = _table([[P(a, "box"), P(f"<b>{b}</b>", "box")] for a, b in facts], [80 * mm, 30 * mm], st,
                   header=False)
    story += [cover, Spacer(1, 25 * mm)]
    story.append(P(f"{_esc(author) + ' · ' if author else ''}généré le {generated:%d/%m/%Y à %H:%M}, "
                   "directement depuis les journaux du logiciel. Aucun chiffre n'est saisi à la main.", "small"))
    story.append(PageBreak())

    # ------------------------------------------------------------- 1. summary
    story.append(P("1. Synthèse", "h1"))
    story.append(P(
        f"Le logiciel a essayé <b>{n_trials} configurations</b> de stratégies sur {len(journals)} univers "
        f"d'investissement. Chacune a été jugée sur la période de construction seulement. "
        f"{n_freezes} ont été gelées, puis testées une seule fois sur 2023-2026. "
        f"<b>{n_valid} {'test est propre' if n_valid == 1 else 'tests sont propres'}</b> : la stratégie avait été gelée avant que quiconque ne regarde la "
        f"période de test. Parmi eux, <b>{n_win}</b> apportent de la valeur face à la référence du client."))
    if tests:
        rows = [["Univers · profil", "Stratégie", "Rendement annuel test (stratégie)", "Écart ajusté du risque",
                 "Verdict", "Statut"]]
        for t in tests:
            rows.append([_esc(_context(t)), _esc(t["name"]), _pct(t["oos"]["cagr"]),
                         _pct(t["oos"].get("ce3_gap_vs_1N"), signed=True),
                         VERDICT_TEXT.get(t["oos"]["verdict"], t["oos"]["verdict"]),
                         t["status"].replace("_", " ")])
        story.append(_table(rows, [42 * mm, 30 * mm, 24 * mm, 22 * mm, 25 * mm, width - 143 * mm], st))
    story.append(Spacer(1, 4 * mm))
    story.append(P("Ce qu'il faut retenir", "h2"))
    lessons = [
        "<b>La règle du jeu est fixée avant les résultats.</b> Profils, seuils de robustesse, seuil d'égalité "
        "et critère de jugement sont écrits dans le code et datés par git avant les tests (section 3).",
        "<b>Tout est compté.</b> Chaque configuration essayée, y compris les échecs, est inscrite au journal ; "
        "le test « chance » devient plus sévère à chaque essai supplémentaire (annexe).",
        "<b>Un seul regard sur l'avenir.</b> La période 2023-2026 n'est consultée qu'une fois par stratégie gelée. "
        "Le statut de chaque test dit s'il est propre ou affaibli.",
        f"<b>Une égalité n'est pas une victoire.</b> Un écart inférieur à {_pct(MATERIALITY, digits=2)} par an "
        "est déclaré égalité.",
        "<b>La preuve finale est le temps.</b> Les stratégies retenues sont suivies en temps réel depuis leur "
        "gel ; ce suivi est le seul test que personne n'a pu regarder à l'avance (section 7).",
    ]
    bullets(lessons)
    story.append(PageBreak())

    # ---------------------------------------------------------------- 2. method
    story.append(P("2. Méthode", "h1"))
    story.append(P("Chaque stratégie suit le même chemin, et aucune étape ne peut être sautée :"))
    steps = [["Étape", "Ce qui se passe", "Ce qui l'empêche de tricher"],
             ["1. Explorer", "La stratégie est comparée à la référence sur la période de construction "
              "(jusqu'au 31/12/2022), sur une fenêtre de dates commune à toutes les stratégies.",
              "Le labo ne reçoit physiquement aucun prix postérieur au 31/12/2022. Chaque essai est compté."],
             ["2. Robustesse", "Cinq tests : réglages voisins, frais, fenêtres de 3 ans, bootstrap par blocs, "
              "Sharpe dégonflé ; plus l'analyse de la composition.",
              "Seuils fixés à l'avance ; un résultat qui tient à un seul réglage ou un seul actif est écarté."],
             ["3. Geler", "Les réglages ET le code qui fait tourner la stratégie sont scellés (empreinte).",
              "Une stratégie gelée ne se modifie plus ; un changement de code est détecté."],
             ["4. Tester une fois", "Un seul passage sur 2023-2026, inscrit au journal avec un statut.",
              "Relancer donne « REPEAT » ; une période déjà consultée donne un statut affaibli."],
             ["5. Suivre", "Dès le lendemain du test, la stratégie est rejouée sur les nouveaux prix.",
              "Le premier jour compté est fixé à l'inscription ; chaque point est conservé."]]
    story.append(_table(steps, [24 * mm, 75 * mm, width - 99 * mm], st))
    story.append(P("Le moteur de calcul", "h2"))
    bullets([
        "Prix réels Yahoo Finance, dividendes réinvestis, <b>convertis en euros jour par jour</b> ; "
        "un jour sans prix est retiré, jamais inventé.",
        "Une décision prise à la clôture est exécutée le jour de bourse suivant : la stratégie ne peut pas "
        "utiliser une information qu'elle n'aurait pas eue.",
        "Frais de transaction déduits ; identité comptable vérifiée chaque jour (valeur = titres + cash).",
        "Juge : <b>l'équivalent certain</b> d'un client d'aversion au risque γ (fonction CRRA). Il traduit "
        "rendement et risque en un seul « rendement garanti équivalent » ; plus γ est élevé, plus les pertes pèsent.",
    ])
    story.append(P("Les verdicts et statuts", "h2"))
    vrows = [["Verdict", "Signification"]] + [[k.replace("_", " "), v] for k, v in VERDICT_TEXT.items()]
    srows = [["Statut du test", "Signification"]] + [[k.replace("_", " "), v] for k, v in STATUS_TEXT.items()]
    story.append(_table(vrows, [45 * mm, width - 45 * mm], st))
    story.append(Spacer(1, 3 * mm))
    story.append(_table(srows, [45 * mm, width - 45 * mm], st))
    story.append(PageBreak())

    # ------------------------------------------------------------ 3. rules
    story.append(P("3. Les règles, fixées avant les résultats", "h1"))
    story.append(P("Profils clients", "h2"))
    story.append(P("Une stratégie est jugée pour un client donné, contre le portefeuille simple qu'un conseiller "
                   "lui proposerait sans stratégie. Hors limites, elle est inadaptée quel que soit son rendement."))
    prow = [["Profil", "Référence", "Juge (γ)", "Volatilité max.", "Pire chute tolérée"]]
    for p in _profiles.PROFILES.values():
        prow.append([p.label, f"{_pct(p.equity_share, digits=0)} actions / {_pct(1 - p.equity_share, digits=0)} "
                     "obligations (cash dans le PEA)",
                     f"{p.gamma:g}".replace(".", ","), _pct(p.max_volatility, digits=0), _pct(-p.max_loss, digits=0)])
    story.append(_table(prow, [24 * mm, 66 * mm, 18 * mm, 26 * mm, width - 134 * mm], st))
    story.append(P("Plafonds de volatilité placés dans les classes de l'indicateur de risque PRIIPs "
                   "(SRI 3 : 5-12 %, SRI 4 : 12-20 %). Sans profil, la référence est 1/N et le juge γ = 3.", "small"))
    story.append(P("Robustesse avant gel", "h2"))
    story.append(_table([["Test", "Question", "Seuil"]] + [list(r) for r in ROBUSTNESS_RULES],
                        [24 * mm, width - 48 * mm, 24 * mm], st))
    story.append(P("Une stratégie est gelée si elle réussit au moins 4 tests sur 5 et si son gain ne repose pas sur "
                   "un seul actif (seuil retenu pour la suite : plus de 50 % du gain venant d'un actif = refus). "
                   f"Égalité si l'écart ajusté du risque est inférieur à {_pct(MATERIALITY, digits=2)} par an.", "body"))
    if data["git"]:
        story.append(P("Chronologie (git)", "h2"))
        story.append(P("Chaque étape a été enregistrée avec git, qui date les versions du projet : on peut "
                       "vérifier que les règles précèdent les résultats.", "body"))
        grows = [["Date", "Version", "Étape"]] + [[g["date"], g["commit"], _esc(g["message"])] for g in data["git"]]
        story.append(_table(grows, [30 * mm, 18 * mm, width - 48 * mm], st))
    story.append(PageBreak())

    # -------------------------------------------------------------- 4. universes
    story.append(P("4. Univers et données", "h1"))
    for name in journals:
        u = _universes.get(name)
        story.append(P(_esc(u["libellé"]), "h2"))
        arows = [["Code", "Actif", "Classe", "Devise", "Existe depuis"]]
        for a in _universes.assets(name):
            arows.append([a["ticker"], _esc(a.get("nom", "")), a["classe"], a.get("devise", ""), a.get("depuis", "")])
        story.append(_table(arows, [22 * mm, 72 * mm, 32 * mm, 16 * mm, width - 142 * mm], st))
        bullets([_esc(r) for r in u["remarques"]])
    story.append(PageBreak())

    # ------------------------------------------------------------ 5. journal
    story.append(P("5. Tout ce qui a été essayé", "h1"))
    story.append(P("Chaque ligne est lue dans le journal du labo, qui ne s'efface pas. Une même stratégie jugée "
                   "pour deux profils compte comme deux essais. La liste complète est en annexe."))
    jrows = [["Univers", "Juge", "Essais", "dont « apporte de la valeur »", "Gelées", "Testées"]]
    for name, j in journals.items():
        by_profile: dict = {}
        for t in j["trials"]:
            by_profile.setdefault(t.get("profile"), []).append(t)
        for prof, items in by_profile.items():
            wins = sum(t["is"].get("verdict") == "ADDS_VALUE" for t in items)
            jrows.append([_esc(j["label"].split(" (")[0]), _profile_label(prof), str(len(items)), str(wins),
                          str(sum(f.get("profile") == prof for f in j["freezes"])),
                          str(sum(t.get("profile") == prof for t in j["tests"]))])
    story.append(_table(jrows, [40 * mm, 42 * mm, 16 * mm, 40 * mm, 18 * mm, width - 156 * mm], st))
    story.append(P("Lecture : la plupart des configurations n'apportent rien face à une diversification simple. "
                   "C'est un résultat connu (DeMiguel, Garlappi et Uppal, 2009) ; le logiciel le montre au lieu "
                   "de ne présenter que la meilleure.", "small"))
    story.append(PageBreak())

    # -------------------------------------------------------------- 6. tests
    story.append(P("6. Les tests uniques sur 2023-2026", "h1"))
    story.append(P("Pour chaque stratégie gelée : sa recette, ses chiffres de construction et de test, et les "
                   "courbes de la période de test. Le verdict et l'écart sont ceux inscrits au journal le jour "
                   "du test ; les courbes sont recalculées avec les prix actuels."))
    tmp = path.parent / f".{path.stem}_graphs"
    tmp.mkdir(exist_ok=True)
    for i, t in enumerate(tests, start=1):
        block = [P(f"6.{i} {_esc(t['name'])} — {_esc(_context(t))}", "h2")]
        f = t["freeze"]
        if f:
            block.append(P("<br/>".join(_esc(r) for r in _recipe(f["spec"])), "small"))
        curves = None
        try:
            curves = test_curves(t, root, price_loader, cache_dir)
        except Exception as exc:  # the report must not die for one chart
            block.append(P(f"Courbes indisponibles : {_esc(exc)}", "small"))
        is_m, oos = (f or {}).get("is", {}), t["oos"]
        ref = curves["summary"][curves["reference"]] if curves else {}
        mrows = [["", "Construction (stratégie)", "Test (stratégie)",
                  f"Test ({_esc(curves['reference'])})" if curves else "Test (référence)"]]
        for label, key, fmt in (("Rendement annuel", "cagr", _pct), ("Volatilité", "volatility", _pct),
                                ("Sharpe", "sharpe", _num), ("Pire chute", "max_drawdown", _pct)):
            mrows.append([label, fmt(is_m.get(key)), fmt(oos.get(key)), fmt(ref.get(key)) if ref else "–"])
        mrows.append(["Écart ajusté du risque", _pct(is_m.get("ce3_gap_vs_1N"), True),
                      _pct(oos.get("ce3_gap_vs_1N"), True), ""])
        block.append(_table(mrows, [45 * mm, 40 * mm, 40 * mm, width - 125 * mm], st))
        verdict = VERDICT_TEXT.get(oos["verdict"], oos["verdict"])
        block.append(P(f"<b>Verdict : {verdict}.</b> Statut : {STATUS_TEXT.get(t['status'], t['status'])}. "
                       f"Gelée le {(f or {}).get('at', '?')[:10]} après {(f or {}).get('n_trials', '?')} essais ; "
                       f"testée le {t['tested_at'][:10]}.", "body"))
        if curves:
            img = _chart(curves, tmp / f"test_{i}.png")
            block.append(Image(str(img), width=width, height=width * 4.2 / 7.2))
        story.append(KeepTogether(block))
        story.append(Spacer(1, 4 * mm))
    if not tests:
        story.append(P("Aucun test unique n'a encore été réalisé."))
    story.append(PageBreak())

    # ------------------------------------------------------------ 7. live
    story.append(P("7. Suivi en temps réel", "h1"))
    story.append(P("Après son test, chaque stratégie retenue est rejouée sur les prix publiés depuis son inscription. "
                   "Aucune donnée de ce suivi n'existait au moment du gel : c'est le seul test que personne n'a "
                   f"pu regarder à l'avance. Moins de 12 mois de suivi ne permettent pas de conclure."))
    lrows = [["Stratégie suivie", "Premier jour compté", "Dernier point", "Jours suivis",
              "Rendement stratégie", "Rendement référence"]]
    for item in data["tracked"]:
        r, pts = item["record"], item["points"]
        last = pts[-1] if pts else None
        lrows.append([_esc(track_title(r["nom"])), _fr_date(r["premier jour compté"]),
                      _fr_date(last["derniers prix"]) if last else "aucun point",
                      str(last["jours suivis"]) if last else "0",
                      _pct(last["rendement"], True) if last else "–",
                      _pct(last["rendement 1/N"], True) if last else "–"])
    story.append(_table(lrows, [52 * mm, 26 * mm, 24 * mm, 18 * mm, 26 * mm, width - 146 * mm], st))
    story.append(P("Mise à jour : bouton « Faire le point maintenant » de la page Suivi en temps réel (ou "
                   "<i>python scripts/paper_trading.py point</i>), puis régénérer ce rapport.", "small"))

    # ------------------------------------------------------------- 8. limits
    story.append(P("8. Limites connues", "h1"))
    bullets([
        "<b>Peu de tests, sur moins de 4 ans, dans un marché haussier.</b> Quelques tests propres sont un indice, "
        "pas une preuve ; plusieurs stratégies testées en même temps augmentent le risque qu'une gagne par hasard.",
        "<b>Le cash des stratégies gelées est compté à 0 %.</b> Les taux monétaires étant négatifs de 2015 à "
        "2022, ce choix a légèrement flatté le cash pendant la construction, puis l'a sous-estimé depuis 2023 "
        "(environ 3 % par an). Les nouvelles stratégies peuvent placer leur cash en fonds monétaire (XEON.DE, "
        "€STR) : c'est alors un essai distinct, et la recette l'indique.",
        "<b>ETF américains comme historique long (univers Monde).</b> Un particulier français détiendrait leurs "
        "équivalents UCITS, sur les mêmes indices mais avec d'autres frais.",
        "<b>Clôtures décalées (univers Monde).</b> Paris clôture à 17 h 30, New York à 22 h : le même jour, les "
        "deux prix ne sont pas pris au même moment. En rendements quotidiens, une partie des mouvements communs "
        "passe au lendemain : les actifs américains et européens paraissent moins liés qu'ils ne le sont, et la "
        "volatilité des portefeuilles mixtes est un peu sous-estimée. Le diagnostic client utilise des "
        "rendements hebdomadaires pour cette raison ; le laboratoire garde les rendements quotidiens pour ne pas "
        "modifier les résultats déjà gelés et testés.",
        "<b>PEA : 8 ans d'historique et des ETF très corrélés</b> (0,6 à 0,98) ; émergents et Japon trop récents. "
        "Le gouvernement a renoncé (26/08/2026) à exclure les ETF synthétiques du PEA ; un amendement au budget "
        "2027 reste possible.",
        "<b>Seuils conventionnels.</b> Profils, seuils de robustesse et seuil d'égalité sont des conventions "
        "raisonnables, fixées à l'avance, pas des vérités : d'autres choix donneraient d'autres conclusions.",
        "<b>Les performances passées ne préjugent pas des performances futures.</b> Ce rapport décrit une "
        "méthode de recherche ; il ne constitue pas un conseil en investissement.",
    ])
    story.append(PageBreak())

    # -------------------------------------------------------------- annex
    story.append(P("Annexe — journal complet des essais", "h1"))
    story.append(P("Mesures sur la période de construction. Écart = équivalent certain de la stratégie moins "
                   "celui de la référence, pour le juge indiqué. Les variantes entre parenthèses sont les réglages "
                   "voisins du test de robustesse. Une même ligne peut revenir si le code de calcul a changé "
                   "entre-temps : c'est alors une autre configuration, comptée à part.", "small"))
    for name, j in journals.items():
        story.append(P(_esc(j["label"]), "h2"))
        rows = [["Date", "Stratégie", "Juge", "Rdt", "Vol.", "Sharpe", "Chute", "Écart", "Verdict"]]
        for t in sorted(j["trials"], key=lambda e: e["at"]):
            m = t["is"]
            rows.append([f"{t['at'][8:10]}/{t['at'][5:7]}/{t['at'][2:4]}", _esc(_fr(readable_name(t["name"]))), _profile_label(t.get("profile")).split(" (")[0],
                         _pct(m.get("cagr")), _pct(m.get("volatility")), _num(m.get("sharpe")),
                         _pct(m.get("max_drawdown")), _pct(m.get("ce3_gap_vs_1N"), True),
                         SHORT_VERDICT.get(m.get("verdict"), m.get("verdict") or "")])
        story.append(_table(rows, [15 * mm, 52 * mm, 20 * mm, 13 * mm, 13 * mm, 14 * mm, 15 * mm, 14 * mm,
                                   width - 156 * mm], st))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("DV", 7)
        canvas.setFillColor(colors.HexColor(GREY))
        canvas.drawString(18 * mm, 10 * mm, "Portfolio Assistant — rapport de recherche")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title="Portfolio Assistant — rapport de recherche", author=author or "")
    doc.build(story, onFirstPage=lambda c, d: None, onLaterPages=footer)
    for png in tmp.glob("*.png"):
        png.unlink()
    tmp.rmdir()
    return path
