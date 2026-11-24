// Portfolio Assistant for VS Code: every command runs the software's own command line
// (python -m portfolio_assistant.cli) in a terminal, with the interpreter you chose.
const vscode = require("vscode");
const path = require("path");

let terminal;

function settings() { return vscode.workspace.getConfiguration("portfolioAssistant"); }

async function python() {
  const own = settings().get("python");
  if (own) return own;
  try {                                               // the interpreter picked in the Python extension
    const ext = vscode.extensions.getExtension("ms-python.python");
    if (ext) {
      if (!ext.isActive) await ext.activate();
      const env = ext.exports.environments;
      const active = env && env.getActiveEnvironmentPath && env.getActiveEnvironmentPath();
      if (active && active.path) return active.path;
    }
  } catch (e) { /* falls back below */ }
  return process.platform === "win32" ? "python" : "python3";
}

function project() {
  const own = settings().get("projet");
  if (own) return own;
  const folders = vscode.workspace.workspaceFolders;
  return folders && folders.length ? folders[0].uri.fsPath : process.cwd();
}

const q = s => (process.platform === "win32" ? `"${String(s).replace(/"/g, '""')}"` : `'${String(s).replace(/'/g, "'\\''")}'`);

async function run(args) {
  const py = await python();
  if (!terminal || terminal.exitStatus !== undefined) terminal = vscode.window.createTerminal({ name: "Portfolio Assistant", cwd: project() });
  terminal.show(true);
  terminal.sendText([q(py), "-m", "portfolio_assistant.cli", "--projet", q(project()), ...args].join(" "));
}

async function currentFile() {
  const ed = vscode.window.activeTextEditor;
  if (!ed || ed.document.languageId !== "python") {
    vscode.window.showWarningMessage("Ouvrez d'abord le fichier .py de votre stratégie.");
    return null;
  }
  if (ed.document.isDirty) await ed.document.save();
  return ed.document.uri.fsPath;
}

async function pickUniverse() {
  const choice = await vscode.window.showQuickPick(["pea", "monde", "ucits"], { placeHolder: "Univers", ignoreFocusOut: true });
  return choice || settings().get("univers");
}

function activate(context) {
  const reg = (name, fn) => context.subscriptions.push(vscode.commands.registerCommand(`portfolioAssistant.${name}`, fn));

  reg("nouvelle", async () => {
    const name = await vscode.window.showInputBox({ prompt: "Nom de la stratégie", value: "Ma stratégie" });
    if (!name) return;
    const model = await vscode.window.showQuickPick(["Les 2 plus réguliers sur 6 mois", "Tendance avec refuge", "Volatilité cible 10 %"],
      { placeHolder: "Modèle de départ" });
    const slug = name.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
    await run(["nouvelle", q(name), ...(model ? ["--modele", q(model)] : [])]);
    const file = path.join(project(), "laboratoire", "code", `${slug}.py`);
    setTimeout(async () => {
      try { await vscode.window.showTextDocument(await vscode.workspace.openTextDocument(file)); } catch (e) { /* not written yet */ }
    }, 2500);
  });
  reg("essayer", async () => {
    const f = await currentFile(); if (!f) return;
    await run(["essayer", q(f), "--univers", settings().get("univers"), "--frais", q(settings().get("frais"))]);
  });
  reg("etat", async () => {
    const f = await currentFile(); if (!f) return;
    const when = await vscode.window.showInputBox({ prompt: "Jusqu'au (AAAA-MM-JJ, avant 2023)", value: "2018-12-31" });
    if (!when) return;
    const state = f.replace(/\.py$/, ".etat.json");
    await run(["essayer", q(f), "--univers", await pickUniverse(), "--frais", q(settings().get("frais")), "--jusqu-au", when, "--etat", q(state)]);
  });
  reg("reprendre", async () => {
    const f = await currentFile(); if (!f) return;
    await run(["reprendre", q(f), "--etat", q(f.replace(/\.py$/, ".etat.json"))]);
  });
  reg("explorer", async () => {
    const f = await currentFile(); if (!f) return;
    const ok = await vscode.window.showWarningMessage("Passer au laboratoire compte un essai au registre. Continuer ?", { modal: true }, "Compter l'essai");
    if (ok) await run(["explorer", q(f), "--univers", await pickUniverse()]);
  });
  reg("geler", async () => {
    const f = await currentFile(); if (!f) return;
    const ok = await vscode.window.showWarningMessage("Une stratégie gelée ne peut plus changer sous ce nom. Geler ?", { modal: true }, "Geler");
    if (ok) await run(["geler", q(f), "--univers", await pickUniverse()]);
  });
  reg("valider", async () => {
    const name = await vscode.window.showInputBox({ prompt: "Nom de la stratégie gelée" });
    if (!name) return;
    await run(["valider", q(name), "--univers", await pickUniverse()]);
  });
  reg("essais", () => run(["essais"]));
  reg("ouvrir", () => vscode.env.openExternal(vscode.Uri.parse(settings().get("adresse"))));

  const item = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
  item.text = "$(play) Essayer";
  item.tooltip = "Portfolio Assistant : essayer cette stratégie dans l'atelier";
  item.command = "portfolioAssistant.essayer";
  const toggle = ed => (ed && ed.document.languageId === "python" ? item.show() : item.hide());
  toggle(vscode.window.activeTextEditor);
  context.subscriptions.push(item, vscode.window.onDidChangeActiveTextEditor(toggle));
}

function deactivate() {}

module.exports = { activate, deactivate };
