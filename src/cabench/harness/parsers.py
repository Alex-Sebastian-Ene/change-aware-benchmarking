"""Versioned answer parsers for generative tasks.

Never edit a registered version in place. Change behaviour = register a new version.
"""

from collections.abc import Callable

Parser = Callable[[str], str]

PARSERS: dict[tuple[str, str], Parser] = {}


def register(task: str, version: str):
    """Decorator: @register("gsm8k", "v1") adds the function to PARSERS."""

    def deco(fn: Parser) -> Parser:
        key = (task, version)
        if key in PARSERS:
            raise ValueError(f"parser {key} already registered, add a new version instead")
        PARSERS[key] = fn
        return fn

    return deco


def get_parser(task: str, version: str) -> Parser:
    """Look up a parser; raise KeyError with a helpful message if missing."""
    try:
        return PARSERS[(task, version)]
    except KeyError:
        known = sorted(v for t, v in PARSERS if t == task)
        raise KeyError(f"no parser for {task!r} {version!r}, known versions: {known}") from None
