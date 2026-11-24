"""Where everything lives in the software's folder (one place: change the layout here only).

    laboratoire/        the research: registries (research_*), strategies, real-time tracking, reports
    donnees_cabinet/    what belongs to this installation and never leaves it: clients, models,
                        universes, contracts, funds, daily backups, price cache (not in git, not in the package)
"""

from __future__ import annotations

import os
from pathlib import Path

LAB = "laboratoire"
CABINET = "donnees_cabinet"


def lab(root: str | Path = ".") -> Path:
    return Path(root) / LAB


def cabinet(root: str | Path = ".") -> Path:
    """PA_DONNEES moves it elsewhere (online: the server's persistent disk)."""
    moved = os.environ.get("PA_DONNEES")
    return Path(moved) if moved else Path(root) / CABINET


def cache(root: str | Path = ".") -> Path:
    return cabinet(root) / "cache"


def journal(name: str) -> str:
    """The registry folder of a universe, relative to the software's folder."""
    return f"{LAB}/research_{name}"
