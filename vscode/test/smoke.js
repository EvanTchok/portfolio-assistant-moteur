// Runs the extension against a fake "vscode" module and checks what reaches the terminal.
const Module = require("module");
const sent = [], commands = {};
const fake = {
  workspace: { getConfiguration: () => ({ get: k => ({ python: "/opt/py/bin/python", projet: "/proj", univers: "pea", frais: "Courtier en ligne", adresse: "http://localhost:8501" })[k] }),
               workspaceFolders: [{ uri: { fsPath: "/proj" } }], openTextDocument: async f => f },
  window: { createTerminal: () => ({ show() {}, sendText: t => sent.push(t), exitStatus: undefined }),
            activeTextEditor: { document: { languageId: "python", isDirty: false, uri: { fsPath: "/proj/laboratoire/code/ma strat.py" } } },
            showQuickPick: async () => "monde", showInputBox: async o => (o.prompt.startsWith("Jusqu") ? "2018-12-31" : "Ma stratégie"),
            showWarningMessage: async (m, o, b) => b, showTextDocument: async () => {}, createStatusBarItem: () => ({ show() {}, hide() {} }),
            onDidChangeActiveTextEditor: () => ({}) },
  commands: { registerCommand: (n, f) => { commands[n] = f; return {}; } },
  extensions: { getExtension: () => null }, env: { openExternal: () => {} }, Uri: { parse: x => x },
  StatusBarAlignment: { Left: 1 },
};
const load = Module._load;
Module._load = function (req, ...rest) { return req === "vscode" ? fake : load.call(this, req, ...rest); };
const ext = require("../extension.js");
ext.activate({ subscriptions: [] });
(async () => {
  for (const c of ["essayer", "etat", "reprendre", "explorer", "geler", "essais"]) await commands[`portfolioAssistant.${c}`]();
  const ok = [
    /^'\/opt\/py\/bin\/python' -m portfolio_assistant.cli --projet '\/proj' essayer '\/proj\/laboratoire\/code\/ma strat.py' --univers pea --frais 'Courtier en ligne'$/,
    /essayer .* --univers monde .* --jusqu-au 2018-12-31 --etat '\/proj\/laboratoire\/code\/ma strat.etat.json'$/,
    /reprendre '\/proj\/laboratoire\/code\/ma strat.py' --etat '\/proj\/laboratoire\/code\/ma strat.etat.json'$/,
    /explorer '.*' --univers monde$/, /geler '.*' --univers monde$/, /essais$/,
  ];
  sent.forEach((s, i) => { if (!ok[i].test(s)) { console.error("MISMATCH", i, s); process.exit(1); } });
  console.log(`${sent.length} commandes vérifiées`);
})();
