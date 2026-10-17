"""
item_id = first 16 hex chars of SHA-256 over benchmark | benchmark_version | question | choice_1 | ... | choice_n
after normalising question and choices (Unicode NFC, collapse whitespace, strip).
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Iterable, Sequence

SEP = "|"
_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    """
    Args:
        text (str): Text to normalise (non-strings are converted with str()).

    Returns:
        str: The normalised text.
    """
    return _WS.sub(" ", unicodedata.normalize("NFC", str(text))).strip()


def item_key(benchmark: str, benchmark_version: str, question: str, choices: Sequence[str] | None = None) -> str:
    """
    Args:
        benchmark (str): Benchmark name.
        benchmark_version (str): Benchmark version.
        question (str): Question text (normalised before joining).
        choices (Sequence[str] | None): Answer choices in the order shown to the
            model (each normalised); None or empty for free-response items.

    Returns:
        str: The parts joined with SEP.
    """
    parts = [str(benchmark), str(benchmark_version), normalise(question)]
    parts += [normalise(c) for c in (choices or [])]
    return SEP.join(parts)


def make_item_id(benchmark: str, benchmark_version: str, question: str, choices: Sequence[str] | None = None) -> str:
    """
    Args:
        benchmark (str): Benchmark name.
        benchmark_version (str): Benchmark version.
        question (str): Question text.
        choices (Sequence[str] | None): Answer choices in the order shown to the
            model; None or empty for free-response items.

    Returns:
        str: First 16 hex characters of the SHA-256 of item_key(...).
    """
    key = item_key(benchmark, benchmark_version, question, choices)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def find_collisions(keys_and_ids: Iterable[tuple[str, str]]) -> dict[str, list[str]]:
    """
    Args:
        keys_and_ids (Iterable[tuple[str, str]]): (item_key, item_id) pairs.

    Returns:
        dict[str, list[str]]: Colliding item_ids mapped to their sorted distinct
        keys; empty if there are no collisions.
    """
    seen: dict[str, set[str]] = {}
    for key, iid in keys_and_ids:
        seen.setdefault(iid, set()).add(key)
    return {iid: sorted(k) for iid, k in seen.items() if len(k) > 1}
