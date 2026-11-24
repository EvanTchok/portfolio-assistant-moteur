"""Strategies written in code: same lab, same counting, same seal; never written online."""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest

from portfolio_assistant import code_rules, services, universes


def _prices(name, cache_dir=None, verbose=False):
    assets = universes.assets(name)
    rng = np.random.default_rng(len(name))
    idx = pd.bdate_range("2012-01-02", "2026-09-30")
    rets = rng.normal(3e-4, 0.01, (len(idx), len(assets)))
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=[a["ticker"] for a in assets])


@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.delenv("PA_EN_LIGNE", raising=False)
    return services.Workspace(tmp_path, price_loader=_prices)


def test_the_examples_compile_and_mistakes_are_explained():
    for source in code_rules.EXAMPLES.values():
        assert callable(code_rules.compile_rule(source))
    with pytest.raises(code_rules.CodeError, match="non autorisé"):
        code_rules.compile_rule("import os\ndef decide(ctx):\n    return {}\n")
    with pytest.raises(code_rules.CodeError, match="decide"):
        code_rules.compile_rule("x = 1\n")
    with pytest.raises(code_rules.CodeError, match="syntaxe"):
        code_rules.compile_rule("def decide(ctx)\n    return {}\n")
    with pytest.raises(code_rules.CodeError):
        code_rules.compile_rule("open('/etc/passwd')\ndef decide(ctx):\n    return {}\n")


def test_code_is_explored_frozen_and_tested_like_any_rule(ws):
    rule = services.code_save(ws, "Réguliers", code_rules.TEMPLATE)
    assert rule == "code:reguliers"
    first = services.code_explore(ws, "monde", None, "Réguliers", rule)
    again = services.code_explore(ws, "monde", None, "Réguliers", rule)
    assert first["new_trial"] and not again["new_trial"] and again["n_trials"] == 1
    services.code_freeze(ws, "monde", None, "Réguliers", rule)
    assert services.test(ws, "Réguliers", "monde", None)["status"] == "VALID_OOS"
    with pytest.raises(ValueError, match="1/N"):
        services.code_explore(ws, "monde", "prudent", "Réguliers", rule)


def test_an_edited_frozen_file_is_caught(ws):
    rule = services.code_save(ws, "Tendance", code_rules.EXAMPLES["Tendance avec refuge"])
    services.code_freeze(ws, "monde", None, "Tendance", rule)
    path = code_rules.folder(ws.lab_dir) / "tendance.py"
    path.write_text(path.read_text(encoding="utf-8").replace("200", "150"), encoding="utf-8")
    assert services.test(ws, "Tendance", "monde", None)["status"] == "CONTAMINATED_CODE_CHANGED"


def test_the_workshop_runs_code_without_touching_the_registry(ws):
    out = services.atelier_run(ws, "monde", None, {"kind": "code", "source": code_rules.TEMPLATE, "name": "x"},
                               {"frais": "Courtier en ligne"})
    assert out["n_configurations"] == 1 and ws.lab("monde", None).n_trials == 0


def test_no_code_is_written_online(ws, monkeypatch):
    monkeypatch.setenv("PA_EN_LIGNE", "1")
    assert not code_rules.editor_allowed() and not code_rules.assistant_ready()
    with pytest.raises(PermissionError):
        services.code_save(ws, "x", code_rules.TEMPLATE)


def test_the_assistant_needs_the_users_own_key_and_returns_only_code(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(code_rules.CodeError, match="ANTHROPIC_API_KEY"):
        code_rules.ask_assistant("trois ETF", ["A", "B"])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-for-the-mock")
    seen = {}

    def fake(req, timeout):
        seen["headers"], seen["body"] = dict(req.header_items()), json.loads(req.data)
        reply = {"content": [{"type": "text", "text": "Voici :\n```python\ndef decide(ctx):\n    return {}\n```"}]}
        return io.BytesIO(json.dumps(reply).encode())

    class Ctx(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(code_rules.urllib.request, "urlopen", lambda req, timeout: Ctx(fake(req, timeout).read()))
    code = code_rules.ask_assistant("rien", ["A"])
    assert code == "def decide(ctx):\n    return {}\n"
    assert seen["body"]["system"] == code_rules.ASSISTANT_SYSTEM and "A" in seen["body"]["messages"][0]["content"]
