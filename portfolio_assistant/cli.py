"""Command line: the workshop and the lab from a terminal (and from the VS Code extension).

    python -m portfolio_assistant.cli nouvelle "Ma stratégie"
    python -m portfolio_assistant.cli essayer laboratoire/code/ma-strategie.py --univers pea --frais "Courtier en ligne"
    python -m portfolio_assistant.cli essayer ma.py --jusqu-au 2018-12-31 --etat etat.json
    python -m portfolio_assistant.cli reprendre ma.py --etat etat.json
    python -m portfolio_assistant.cli explorer ma.py --univers pea        # counted as a trial
    python -m portfolio_assistant.cli geler ma.py --univers pea
    python -m portfolio_assistant.cli valider "Ma stratégie" --univers pea
    python -m portfolio_assistant.cli essais

Same rules as the pages: the workshop only sees the construction period, ``explorer`` counts
a trial, ``geler`` seals, nothing here opens the sealed period. ``--json`` prints a machine-
readable result (used by the extension).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def _ws(root: str):
    from . import services
    return services.Workspace(root)


def _name(path: Path) -> str:
    return path.stem.replace("-", " ").replace("_", " ").strip().capitalize() or "Stratégie"


def _print_table(rows: dict[str, dict]) -> None:
    labels = list(rows)
    keys = list(next(iter(rows.values())))
    width = max(len(k) for k in keys) + 2
    print(" " * width + "".join(f"{l[:34]:>36}" for l in labels))
    for k in keys:
        cells = []
        for l in labels:
            v = rows[l][k]
            cells.append(f"{v:>36.2f}" if k == "Sharpe" else f"{v * 100:>35.1f}%")
        print(f"{k:<{width}}" + "".join(cells))


def cmd_nouvelle(a) -> dict:
    from . import code_rules
    ws = _ws(a.projet)
    folder = code_rules.folder(ws.lab_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{code_rules.slug(a.nom)}.py"
    if path.exists():
        raise SystemExit(f"{path} existe déjà")
    path.write_text(code_rules.EXAMPLES.get(a.modele, code_rules.TEMPLATE), encoding="utf-8")
    return {"fichier": str(path)}


def cmd_essayer(a) -> dict:
    from . import code_rules, execution as ex, services
    path = Path(a.fichier)
    source = path.read_text(encoding="utf-8")
    ws = _ws(a.projet)
    if a.jusqu_au or a.etat:
        decide = code_rules.compile_rule(source)
        lab = ws.lab(a.univers, None)
        prices = lab.is_prices.loc[:a.jusqu_au] if a.jusqu_au else lab.is_prices
        cfg = ex.AdvancedConfig(costs=ex.COST_PRESETS[a.frais], leverage=a.levier, max_gross=max(a.levier, 1.0),
                                rebalance=a.revision)
        res = ex.run_advanced(prices, decide, cfg, label=_name(path))
        if a.etat:
            Path(a.etat).write_text(json.dumps({"fichier": str(path), "univers": a.univers, "frais": a.frais,
                                                "levier": a.levier, "revision": a.revision,
                                                "empreinte": code_rules.fingerprint(source),
                                                "etat": res.state.to_dict()}, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
        from .metrics import summary
        s = summary(res.nav)
        out = {"période": [str(prices.index[0].date()), str(prices.index[-1].date())], "valeur finale": float(res.nav.iloc[-1]),
               "rendement annuel": s["cagr"], "volatilité": s["volatility"], "Sharpe": s["sharpe"],
               "pire chute": s["max_drawdown"], "état": a.etat}
        if not a.json:
            print(f"{_name(path)} — {out['période'][0]} → {out['période'][1]} : {s['cagr']:.1%} par an, volatilité "
                  f"{s['volatility']:.1%}, Sharpe {s['sharpe']:.2f}, pire chute {s['max_drawdown']:.1%}"
                  + (f" ; état sauvegardé dans {a.etat}" if a.etat else ""))
        return out
    out = services.atelier_run(ws, a.univers, None, {"kind": "code", "source": source, "name": _name(path),
                                                     "rebalance": a.revision},
                               {"frais": a.frais, "levier": a.levier})
    if not a.json:
        print(f"{_name(path)} — atelier, période de construction {out['period'][0]} → {out['period'][1]}")
        _print_table(out["metrics"].to_dict())
        if not out["costs"].empty:
            print("\nFrais et financement :")
            for _, r in out["costs"].iterrows():
                print(f"  {r['poste']:<40}{r['euros']:>12,.0f} €")
        print(f"\nConfigurations essayées dans l'atelier : {out['n_configurations']}")
    return {"métriques": out["metrics"].to_dict(), "période": out["period"], "configurations": out["n_configurations"]}


def cmd_reprendre(a) -> dict:
    from . import code_rules, execution as ex
    saved = json.loads(Path(a.etat).read_text(encoding="utf-8"))
    path = Path(a.fichier or saved["fichier"])
    source = path.read_text(encoding="utf-8")
    if code_rules.fingerprint(source) != saved["empreinte"]:
        raise SystemExit("le fichier a changé depuis la sauvegarde de l'état : la reprise ne serait pas le même calcul")
    ws = _ws(a.projet)
    lab = ws.lab(saved["univers"], None)
    prices = lab.is_prices.loc[:a.jusqu_au] if a.jusqu_au else lab.is_prices
    cfg = ex.AdvancedConfig(costs=ex.COST_PRESETS[saved["frais"]], leverage=saved["levier"],
                            max_gross=max(saved["levier"], 1.0), rebalance=saved["revision"])
    res = ex.run_advanced(prices, code_rules.compile_rule(source), cfg, resume=ex.RunState.from_dict(saved["etat"]))
    saved["etat"] = res.state.to_dict()
    Path(a.etat).write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")
    out = {"de": str(res.nav.index[0].date()), "à": str(res.nav.index[-1].date()), "valeur finale": float(res.nav.iloc[-1])}
    if not a.json:
        print(f"Repris du {out['de']} au {out['à']} : valeur {out['valeur finale']:,.0f} € ; état mis à jour dans {a.etat}")
    return out


def cmd_explorer(a) -> dict:
    from . import services
    path = Path(a.fichier)
    ws = _ws(a.projet)
    rule = services.code_save(ws, _name(path), path.read_text(encoding="utf-8"))
    out = services.code_explore(ws, a.univers, None, _name(path), rule, a.revision)
    if not a.json:
        print(f"{_name(path)} — laboratoire ({'nouvel essai compté' if out['new_trial'] else 'déjà essayée, non recomptée'}"
              f", {out['n_trials']} essais au registre)\nVerdict : {out['verdict']} — {out['reason']}")
        _print_table(out["metrics"].to_dict())
    return {"verdict": out["verdict"], "essais": out["n_trials"], "nouvel essai": out["new_trial"],
            "métriques": out["metrics"].to_dict()}


def cmd_geler(a) -> dict:
    from . import services
    path = Path(a.fichier)
    ws = _ws(a.projet)
    rule = services.code_save(ws, _name(path), path.read_text(encoding="utf-8"))
    rec = services.code_freeze(ws, a.univers, None, _name(path), rule, a.revision)
    if not a.json:
        print(f"{_name(path)} gelée (empreinte {rec['fingerprint']}). Testez-la une seule fois depuis la page Tester.")
    return {"empreinte": rec["fingerprint"]}


def cmd_valider(a) -> dict:
    from . import services
    ws = _ws(a.projet)
    out = services.validate(ws, a.univers, a.profil, name=a.nom, n_baskets=a.paniers)
    if not a.json:
        for t in out["report"]["tests"]:
            print(f"{t['n']:>2}. {t['test']:<24}{t['verdict']:<16}{t['phrase']}")
        s = out["report"]["synthèse"]
        print(f"\n{s['passés']} passés, {s['limites']} limites, {s['échoués']} échoués, {s['non applicables']} non joués.")
    return out["report"]


def cmd_essais(a) -> dict:
    from . import services, universes
    ws = _ws(a.projet)
    out = {}
    for u in universes.UNIVERSES:
        try:
            out[u] = ws.lab(u, None).n_trials
        except Exception:
            continue
    if not a.json:
        for u, n in out.items():
            print(f"{u:<8}{n:>5} essais comptés")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="portfolio-assistant", description="Atelier et laboratoire en ligne de commande")
    p.add_argument("--projet", default=".", help="dossier du logiciel (par défaut : le dossier courant)")
    p.add_argument("--json", action="store_true", help="résultat en JSON")
    sub = p.add_subparsers(dest="commande", required=True)

    def common(s, fichier=True):
        if fichier:
            s.add_argument("fichier")
        s.add_argument("--univers", default="pea")
        s.add_argument("--revision", default="monthly", choices=["weekly", "monthly", "quarterly", "yearly"])

    s = sub.add_parser("nouvelle", help="créer une stratégie en code à partir d'un modèle")
    s.add_argument("nom")
    s.add_argument("--modele", default="")
    s = sub.add_parser("essayer", help="essayer dans l'atelier (période de construction)")
    common(s)
    s.add_argument("--frais", default="Courtier en ligne")
    s.add_argument("--levier", type=float, default=1.0)
    s.add_argument("--jusqu-au", dest="jusqu_au", default=None)
    s.add_argument("--etat", default=None, help="sauvegarder l'état final pour reprendre plus tard")
    s = sub.add_parser("reprendre", help="reprendre un calcul sauvegardé sur les séances suivantes")
    s.add_argument("fichier", nargs="?")
    s.add_argument("--etat", required=True)
    s.add_argument("--jusqu-au", dest="jusqu_au", default=None)
    for name, helptext in (("explorer", "passer au laboratoire (compte comme un essai)"), ("geler", "geler la stratégie")):
        common(sub.add_parser(name, help=helptext))
    s = sub.add_parser("valider", help="les 12 tests de validation d'une stratégie gelée")
    s.add_argument("nom")
    s.add_argument("--univers", default="pea")
    s.add_argument("--profil", default=None)
    s.add_argument("--paniers", type=int, default=40)
    sub.add_parser("essais", help="nombre d'essais comptés par univers")
    a = p.parse_args(argv)
    out = globals()[f"cmd_{a.commande}"](a)
    if a.json:
        print(json.dumps(out, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
