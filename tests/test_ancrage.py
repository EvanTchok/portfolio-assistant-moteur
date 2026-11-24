"""Anchors: an appended line keeps the registry valid, a removed or edited one is caught."""

from __future__ import annotations

from portfolio_assistant import ancrage


def test_appending_is_fine_rewriting_is_caught(tmp_path):
    lab = tmp_path / "laboratoire"
    reg = lab / "research_pea" / "registry.jsonl"
    reg.parent.mkdir(parents=True)
    reg.write_text('{"event": "trial", "n": 1}\n{"event": "trial", "n": 2}\n', encoding="utf-8")
    chain = tmp_path / "ancrages.jsonl"
    first = ancrage.anchor(lab, chain)
    assert first["fichiers"]["research_pea/registry.jsonl"]["lignes"] == 2
    assert ancrage.anchor(lab, chain) is None                                # nothing new
    with reg.open("a", encoding="utf-8") as fh:
        fh.write('{"event": "freeze"}\n')
    second = ancrage.anchor(lab, chain)
    assert second["précédent"] and ancrage.verify(lab, chain)["ok"]
    reg.write_text('{"event": "trial", "n": 1}\n{"event": "freeze"}\n', encoding="utf-8")   # trial 2 removed
    v = ancrage.verify(lab, chain)
    assert not v["ok"] and any("ont changé" in p or "il n'en reste" in p for p in v["problèmes"])


def test_a_broken_chain_is_caught(tmp_path):
    lab = tmp_path / "laboratoire"
    (lab / "suivi").mkdir(parents=True)
    (lab / "suivi" / "journal.jsonl").write_text("a\n", encoding="utf-8")
    chain = tmp_path / "c.jsonl"
    ancrage.anchor(lab, chain)
    (lab / "suivi" / "journal.jsonl").write_text("a\nb\n", encoding="utf-8")
    ancrage.anchor(lab, chain)
    lines = chain.read_text(encoding="utf-8").splitlines()
    first = __import__("json").loads(lines[0])
    first["date"] = "2020-01-01T00:00:00+00:00"                              # re-dated afterwards
    chain.write_text(__import__("json").dumps(first, sort_keys=True) + "\n" + lines[1] + "\n", encoding="utf-8")
    assert any("chaînage" in p for p in ancrage.verify(lab, chain)["problèmes"])
