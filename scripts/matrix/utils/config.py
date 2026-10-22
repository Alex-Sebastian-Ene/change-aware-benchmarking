"""Pinned configs and config_id

config_id = first 16 hex chars of SHA-256 of the raw bytes of the YAML file.
Hashing raw bytes (not parsed YAML) means *any* edit, even a comment, gives a
new ID, which matches the rule that config files are never edited in place.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

REQUIRED_KEYS = {
    "harness", "tasks", "decoding", "backend", "environment", "chat_template",
}
UNPINNED_MARKERS = ("TBD", "TODO", "FILL_ME")


def config_id(path: str | Path) -> str:
    """
    Args:
        path (str | Path): The YAML config file.

    Returns:
        str: First 16 hex characters of the file's SHA-256.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def load_config(path: str | Path, allow_unpinned: bool = False) -> dict:
    """
    Args:
        path (str | Path): The YAML config file.
        allow_unpinned (bool): If True, skip the check for TBD/TODO/FILL_ME markers.

    Returns:
        dict: The parsed config, with its ID added under "_config_id".
    """
    text = Path(path).read_text()
    cfg = yaml.safe_load(text)
    missing = REQUIRED_KEYS - cfg.keys()
    if missing:
        raise ValueError(f"{path}: missing sections {sorted(missing)}")
    if not allow_unpinned:
        hits = [m for m in UNPINNED_MARKERS if m in text]
        if hits:
            raise ValueError(f"{path}: still contains {hits}; pin every value before running")
    cfg["_config_id"] = config_id(path)
    return cfg


def register(path: str | Path, registry: str | Path) -> str:
    """
    Args:
        path (str | Path): The YAML config file to register.
        registry (str | Path): Tab-separated "config_id<TAB>filename" file; created if absent.

    Returns:
        str: The config's config_id.
    """
    path, registry = Path(path), Path(registry)
    cid = config_id(path)
    rows = {}
    if registry.exists():
        for line in registry.read_text().splitlines():
            if line.strip():
                k, v = line.split("\t")
                rows[v] = k
    if path.name in rows and rows[path.name] != cid:
        raise RuntimeError(
            f"{path.name} was edited in place (registered {rows[path.name]}, now {cid}). "
            "Restore it and copy to a new file instead.")
    if path.name not in rows:
        with registry.open("a") as f:
            f.write(f"{cid}\t{path.name}\n")
    return cid
