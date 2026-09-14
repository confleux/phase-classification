"""Canonical CholecT50 label-head definitions shared across the package.

The vendored loader returns per-frame label tuples in a fixed order; losses,
metrics, the CLI, and the config schema must all agree on it. This module is
the single source of truth for the head names, their position in the label
tuple, which heads are single-label, and the default per-head class counts.
"""

from __future__ import annotations

from typing import Literal

HeadName = Literal["phase", "tool", "verb", "target", "triplet"]

# Position of each head in the label tuple returned by the vendored loader.
HEAD_TUPLE_INDEX: dict[str, int] = {
    "triplet": 0,
    "tool": 1,
    "verb": 2,
    "target": 3,
    "phase": 4,
}

# Heads where a frame normally has exactly one active class (CE); the rest are
# multi-label binary vectors (BCE).
SINGLE_LABEL_HEADS: tuple[str, ...] = ("phase", "verb")

# CholecT50 label space. Numbers are dataset facts, not architecture facts.
DEFAULT_NUM_CLASSES: dict[str, int] = {
    "phase": 7,
    "tool": 6,
    "verb": 10,
    "target": 15,
    "triplet": 100,
}


def head_index(name: str) -> int:
    """Return the position of ``name`` in the loader's label tuple."""
    return HEAD_TUPLE_INDEX[name]
