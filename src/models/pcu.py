"""PCU weight lookup keyed by DriveIndia class ID.

Every lookup goes through the integer class id; names are cosmetic
(reporting only) and never take part in a calculation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "pcu_weights.yaml"

_cache: dict | None = None


def _config() -> dict:
    global _cache
    if _cache is None:
        with open(CONFIG, encoding="utf-8") as f:
            _cache = yaml.safe_load(f) or {}
    return _cache


def weight(class_id: int) -> float | None:
    """PCU for a class id, or None when the class is not density-relevant."""
    return _config().get("by_id", {}).get(int(class_id))


def label(class_id: int) -> str:
    """Display label for a PCU class id (reporting only)."""
    return _config().get("labels", {}).get(int(class_id), f"id{int(class_id)}")


def class_ids() -> list[int]:
    """The PCU-relevant class ids."""
    return sorted(int(k) for k in _config().get("by_id", {}))


def total(class_ids_: Iterable[int]) -> float:
    """Sum PCU over an iterable of class ids; irrelevant ids contribute 0."""
    return sum(w for cid in class_ids_ if (w := weight(cid)) is not None)
