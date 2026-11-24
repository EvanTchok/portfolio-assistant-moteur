# Portfolio Assistant pour VS Code

Écrire une stratégie en Python, l'essayer dans l'atelier, la passer au laboratoire (où elle compte comme un essai), la geler et lancer les 12 tests de validation, sans quitter l'éditeur.

## Commandes (palette : « Portfolio Assistant »)

| Commande | Effet |
|---|---|
| Nouvelle stratégie | crée `laboratoire/code/<nom>.py` à partir d'un modèle |
| Essayer dans l'atelier (⌘⌥E / Ctrl+Alt+E, bouton ▶) | frais réels, période de construction seulement, rien n'est compté au laboratoire |
| Essayer jusqu'à une date et sauvegarder l'état | arrête le calcul à une date et garde son état (`<fichier>.etat.json`) |
| Reprendre un calcul sauvegardé | repart de l'état sur les séances suivantes, sans tout recalculer ; refusé si le fichier a changé |
| Passer au laboratoire | compte un essai au registre (confirmation demandée) |
| Geler la stratégie | scelle son code et ses réglages (confirmation demandée) |
| Lancer les 12 tests de validation | sur une stratégie gelée |
| Essais comptés par univers | |

Extraits : `decide`, `pa-momentum`, `pa-volcible`, `pa-parite`.

## Réglages

- `portfolioAssistant.python` : l'interpréteur du logiciel (vide : celui de l'extension Python).
- `portfolioAssistant.projet` : le dossier du logiciel (vide : le dossier ouvert).
- `portfolioAssistant.univers`, `portfolioAssistant.frais` : univers et barème par défaut.

Chaque commande lance `python -m portfolio_assistant.cli` dans un terminal : ce que vous voyez est exactement ce que fait le logiciel.
