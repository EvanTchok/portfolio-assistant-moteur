# Publier le moteur (PyPI) et l'extension (VS Code Marketplace)

Tout est prêt dans ce dépôt ; il ne manque que les comptes, que seul leur propriétaire peut créer.
Aucun mot de passe ni jeton ne doit être écrit dans un fichier du dépôt.

## 1. PyPI : `pip install portfolio-assistant-moteur`

Le paquet se construit et passe `twine check` (`python -m build`). La publication passe par
GitHub Actions (`.github/workflows/publier.yml`) avec la « publication de confiance » de PyPI :
aucun jeton à créer ni à conserver.

1. Créer un compte sur <https://pypi.org/account/register/> et activer la double authentification
   (PyPI l'exige).
2. Sur <https://pypi.org/manage/account/publishing/>, « Add a new pending publisher » :
   - PyPI Project Name : `portfolio-assistant-moteur`
   - Owner : `EvanTchok` — Repository name : `portfolio-assistant-moteur`
   - Workflow name : `publier.yml` — Environment name : `pypi`
3. Sur GitHub, dans ce dépôt : Settings → Environments → New environment → `pypi`
   (on peut y exiger sa propre validation avant chaque publication).
4. Publier une version : onglet Releases → « Draft a new release » → étiquette `v0.1.0` →
   « Publish release ». Le workflow construit et publie ; la page
   <https://pypi.org/project/portfolio-assistant-moteur/> apparaît en une minute.
5. Pour une version suivante : augmenter `version` dans `pyproject.toml`, puis une nouvelle release.

## 2. VS Code Marketplace

Le fichier `vscode/portfolio-assistant-0.1.0.vsix` est déjà construit ; l'éditeur déclaré dans
`vscode/package.json` est `evan-peronnier`.

1. Se connecter avec un compte Microsoft sur <https://marketplace.visualstudio.com/manage> et créer
   l'éditeur (publisher) dont l'identifiant est exactement `evan-peronnier`.
2. Le plus simple : sur cette même page, « New extension » → « Visual Studio Code » → déposer le
   fichier `.vsix`. Rien d'autre à installer.
3. Ou en ligne de commande : créer dans Azure DevOps (<https://dev.azure.com>) un jeton d'accès
   personnel avec la portée « Marketplace → Manage », puis dans `vscode/` :
   `npx @vscode/vsce login evan-peronnier` (le jeton est demandé à l'écran, jamais écrit dans un
   fichier) et `npx @vscode/vsce publish`.
4. Optionnel, pour VSCodium et Cursor : le même `.vsix` sur <https://open-vsx.org> (compte GitHub,
   jeton créé sur open-vsx.org, `npx ovsx publish portfolio-assistant-0.1.0.vsix -p <jeton>`).

## Avant de publier

- Relire `README.md` : c'est la page d'accueil sur PyPI.
- Vérifier que le nom `portfolio-assistant-moteur` est libre sur PyPI (la recherche sur pypi.org suffit).
