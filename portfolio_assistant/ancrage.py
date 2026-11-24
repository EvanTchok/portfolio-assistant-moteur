"""Anchoring the registries: proof that the past of the lab was not rewritten.

The registries only append (research_*/registry.jsonl, suivi/journal.jsonl, atelier/registre.jsonl).
``anchor`` writes, for each one, how many lines it holds and the SHA-256 of those lines, and
chains the entries (each one carries the hash of the previous one). Published where nobody can
quietly change it (the public repository's history), the chain lets anyone who is later shown
the registries check, with ``verify``, that every line anchored on a date is still there,
unchanged and in the same order: a trial cannot be removed, a test cannot be re-dated, a
freeze cannot be edited.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

PATTERNS = ("research_*/registry.jsonl", "suivi/journal.jsonl", "atelier/registre.jsonl")
PUBLIC_URL = "https://github.com/EvanTchok/portfolio-assistant-moteur/blob/main/registre/ancrages.jsonl"


def _lines(path: Path) -> list[bytes]:
    return path.read_bytes().splitlines(keepends=True) if path.exists() else []


def snapshot(lab_dir: str | Path) -> dict[str, dict]:
    lab_dir = Path(lab_dir)
    out = {}
    for pattern in PATTERNS:
        for path in sorted(lab_dir.glob(pattern)):
            lines = _lines(path)
            out[path.relative_to(lab_dir).as_posix()] = {"lignes": len(lines),
                                                          "sha256": hashlib.sha256(b"".join(lines)).hexdigest()}
    return out


def read_chain(chain: str | Path) -> list[dict]:
    path = Path(chain)
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []


def _link(entry: dict) -> str:
    return hashlib.sha256(json.dumps(entry, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def anchor(lab_dir: str | Path, chain: str | Path, *, now: dt.datetime | None = None) -> dict | None:
    """Appends an entry when a registry changed since the last one; returns it (None if nothing new)."""
    entries = read_chain(chain)
    snap = snapshot(lab_dir)
    if entries and entries[-1]["fichiers"] == snap:
        return None
    entry = {"n": len(entries) + 1, "date": (now or dt.datetime.now(dt.timezone.utc)).isoformat(timespec="seconds"),
             "fichiers": snap, "précédent": _link(entries[-1]) if entries else None}
    path = Path(chain)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
    return entry


def verify(lab_dir: str | Path, chain: str | Path) -> dict:
    """Checks the chain and the registries against it; ``ok`` is False with the reasons otherwise."""
    lab_dir = Path(lab_dir)
    entries = read_chain(chain)
    problems = []
    for i, e in enumerate(entries):
        if i and e.get("précédent") != _link(entries[i - 1]):
            problems.append(f"ancrage {e.get('n')} : le chaînage avec l'ancrage précédent est rompu")
        for rel, rec in e["fichiers"].items():
            lines = _lines(lab_dir / rel)
            if len(lines) < rec["lignes"]:
                problems.append(f"{rel} : {rec['lignes']} lignes ancrées le {e['date'][:10]}, il n'en reste que {len(lines)}")
            elif hashlib.sha256(b"".join(lines[:rec["lignes"]])).hexdigest() != rec["sha256"]:
                problems.append(f"{rel} : les {rec['lignes']} premières lignes ont changé depuis l'ancrage du {e['date'][:10]}")
    return {"ok": not problems and bool(entries), "ancrages": len(entries), "problèmes": problems,
            "dernier": entries[-1]["date"] if entries else None}
