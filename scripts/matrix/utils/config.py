"""Pinned configs and config_id

The rules live in cabench.harness.pins (configs/harness/README.md); this module
gives the matrix scripts the same IDs. config_id = SHA-256 of the YAML file
with CRLF normalised to LF, so Windows and Linux checkouts agree. Hashing the
bytes (not parsed YAML) means *any* edit, even a comment, gives a new ID, and
the filename must end in the first 12 hex chars of that hash, so a config
edited in place no longer loads.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from cabench.harness.pins import config_hash
from cabench.harness.pins import load_config as load_pinned

UNPINNED_MARKERS = ("TBD", "TODO", "FILL_ME")


def config_id(path: str | Path) -> str:
    """
    Args:
        path (str | Path): The YAML config file.

    Returns:
        str: The file's SHA-256 (64 hex chars), as cabench.harness.pins computes it.
    """
    return config_hash(Path(path))


def load_config(path: str | Path, allow_unpinned: bool = False) -> dict:
    """
    Args:
        path (str | Path): The YAML config file.
        allow_unpinned (bool): If True, skip the check for TBD/TODO/FILL_ME markers.

    Returns:
        dict: The parsed config, with its ID added under "_config_id". Raises
        cabench.harness.pins.ConfigError if the file breaks a pinning rule.
    """
    pinned = load_pinned(Path(path))
    text = Path(path).read_text()
    if not allow_unpinned:
        hits = [m for m in UNPINNED_MARKERS if m in text]
        if hits:
            raise ValueError(f"{path}: still contains {hits}; pin every value before running")
    cfg = yaml.safe_load(text)
    cfg["_config_id"] = pinned.config_id
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
