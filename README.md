# Portfolio Assistant — moteur

[![tests](https://github.com/EvanTchok/portfolio-assistant-moteur/actions/workflows/tests.yml/badge.svg)](https://github.com/EvanTchok/portfolio-assistant-moteur/actions/workflows/tests.yml)

*English below.*

Le moteur de recherche de [Portfolio Assistant](https://portfolio-assistant-foqd.onrender.com) : un laboratoire de
stratégies d'allocation et de trading **qui compte chaque essai, ne teste qu'une fois et dit ce qu'il ne sait pas**.
Python pur (numpy, pandas, scipy), sous licence MIT.

## Ce qu'il fait

- **Moteur sans regard vers l'avenir** : la règle ne reçoit que les cours connus à la clôture, l'ordre part à la
  clôture suivante ; la valeur est expliquée chaque jour par les prix, les positions, les intérêts et les frais, ou le
  calcul s'arrête.
- **Laboratoire honnête** : chaque configuration regardée est un essai compté (empreinte des réglages et du code) ; une
  stratégie est **gelée**, puis testée **une seule fois** sur une période jamais vue ; un second regard est marqué.
- **12 tests de validation** : référence, prévision du risque, taille neutre, levier, concentration du gain,
  Monte-Carlo par blocs, pouvoir d'achat (IPCH France), stress inversé, triage, Sharpe dégonflé, hors échantillon,
  paniers multi-actifs.
- **Atelier** : commission avec minimum, écart achat-vente, glissement, impact de marché, TTF historique, droit de
  timbre, change ; levier, vente à découvert, CFD, appels de marge ; frais de gestion et de performance ; déclencheurs à
  chaque séance ; **reprise d'un calcul** depuis un état sauvegardé.
- **Mode algorithmique** : signaux, ordres au marché ou à cours limité, stops et objectifs (en séance avec les plus
  hauts et plus bas), grille de réglages sur plusieurs cœurs, mode Recherche (MFE/MAE, premier touché), entraînement à
  l'aveugle.
- **Stratégies en code** (`decide(ctx)`), en ligne de commande et dans **VS Code** (`vscode/`).

## Démarrer

```bash
pip install -e ".[tests]"
portfolio-assistant nouvelle "Ma stratégie"
portfolio-assistant essayer laboratoire/code/ma-strategie.py --univers monde --frais "Courtier en ligne"
portfolio-assistant explorer laboratoire/code/ma-strategie.py --univers monde     # compte un essai
portfolio-assistant geler laboratoire/code/ma-strategie.py --univers monde
python -m pytest -q
```

```python
from portfolio_assistant import execution as ex
from portfolio_assistant.strategies import momentum_top_k
res = ex.run_advanced(prices, momentum_top_k(126, 2, 5),
                      ex.AdvancedConfig(lookback=130, costs=ex.COST_PRESETS["Courtier en ligne"], leverage=1.5, max_gross=1.5))
print(ex.cost_table(res))
later = ex.run_advanced(more_prices, momentum_top_k(126, 2, 5), cfg, resume=res.state)   # reprend sans tout recalculer
```

Les cours viennent de Yahoo Finance (ou d'EODHD sous licence, `PA_SOURCE_PRIX=eodhd`) et sont gardés en cache.

## Vitesse, mesurée honnêtement

`vitesse/` : mêmes stratégies dans ce moteur, vectorbt et NautilusTrader. Sur 5 000 séances, ce moteur met 0,06 s (10 lignes) à 0,11 s (50 lignes), vectorbt 0,03 à 0,08 s (code compilé, 13 s de compilation au premier appel) ;
les trois donnent les mêmes positions, et ce moteur la même valeur finale que vectorbt à 0,01 % près.

## Registres ancrés

`registre/ancrages.jsonl` publie, à chaque ancrage, le nombre de lignes et l'empreinte SHA-256 de chaque registre du
laboratoire en ligne (essais, gels, tests, suivi, atelier), chaque ancrage portant l'empreinte du précédent. Les
registres eux-mêmes ne sont pas publiés, seulement leurs empreintes. L'historique de ce dépôt date chaque ancrage :
quiconque reçoit plus tard les registres vérifie avec `portfolio-assistant verifier --chaine registre/ancrages.jsonl`
qu'aucun essai n'a été retiré, aucun test redaté, aucun gel modifié depuis.

## Ce qui n'est pas ici

Le logiciel du conseiller (comptes, clients, profil MiFID, documents, fiscalité, paiement) reste dans la version
en ligne. Aucune donnée personnelle n'est publiée, et des registres de recherche seulement les empreintes.

Portfolio Assistant est une aide à la décision : il ne passe aucun ordre et ne constitue pas un conseil en
investissement. Les performances passées ne préjugent pas des performances futures.

---

## English

The research engine of Portfolio Assistant: a lab for allocation and trading strategies that **counts every trial,
tests only once and says what it does not know**. Pure Python, MIT licence.

- No look-ahead engine with a daily accounting identity check.
- Honest lab: every configuration looked at is a counted trial; strategies are frozen, then tested once on unseen data.
- 12 validation tests (reference, risk forecast, neutral sizing, leverage, concentration, block Monte Carlo,
  purchasing power, reverse stress, triage, deflated Sharpe, out of sample, multi-asset baskets).
- Workshop: realistic costs and taxes, leverage, shorts, CFDs, margin calls, fund fees, triggers, run state & resume.
- Algorithmic mode, parameter grid on all cores, research mode, blind practice, strategies in code, CLI and VS Code
  extension.

- Anchored registries: `registre/ancrages.jsonl` publishes, for each registry of the online lab, its line count and
  SHA-256, chained entry to entry; `portfolio-assistant verifier` checks registries shown later against it.

`pip install -e ".[tests]"`, then `portfolio-assistant --help`.
