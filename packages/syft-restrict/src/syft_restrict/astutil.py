"""Pure AST + line-range helpers shared by the verifier, obfuscator, and runner."""

from __future__ import annotations

import ast

from pydantic import BaseModel


# ── line ranges ────────────────────────────────────────────────────────────────────────────
# The caller marks the private region as a list of [start, end] 1-based inclusive line ranges.
def normalize_ranges(private) -> list[tuple[int, int]]:
    """Coerce the caller's ``[[lo, hi], ...]`` into a list of ``(int, int)`` tuples.

    Raises ``ValueError`` on a malformed range (``hi < lo``) rather than silently matching no
    lines -- an inverted range must never be mistaken for "nothing to check here".
    """
    ranges = [(int(lo), int(hi)) for lo, hi in private]
    for lo, hi in ranges:
        if hi < lo:
            raise ValueError(f"invalid range [{lo}, {hi}]: end must be >= start")
    return ranges


def row_in_ranges(row: int, ranges) -> bool:
    """True if a 1-based line number falls inside any range."""
    return any(lo <= row <= hi for lo, hi in ranges)


def node_in_ranges(node: ast.AST, ranges) -> bool:
    """True if a node has a line number and it falls inside any range.

    Some nodes (e.g. ``ast.comprehension``) carry no position; those are never "in range".
    """
    line = getattr(node, "lineno", None)
    return line is not None and row_in_ranges(line, ranges)


# ── reading names and dotted paths off the tree ──────────────────────────────────────────────
def dotted(node: ast.AST) -> str | None:
    """The dotted path for a pure Name/Attribute chain (``jnp.numpy.einsum``), else None.

    Returns None the moment the chain hits anything dynamic (a call, a subscript), because such a
    chain can no longer be resolved to a static import path.
    """
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def is_dunder(name: str) -> bool:
    """True for any identifier that starts with ``__`` (the reflection/hook surface)."""
    return name.startswith("__")


def self_attr_name(node: ast.AST) -> str | None:
    """The attr for a single-level ``self.<name>`` / ``cls.<name>`` access, else None."""
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in ("self", "cls")
    ):
        return node.attr
    return None


def rooted_in_self(node: ast.AST) -> bool:
    """True if an Attribute/Subscript chain's ultimate base is ``self`` / ``cls``."""
    cur = node
    while isinstance(cur, (ast.Attribute, ast.Subscript)):
        cur = cur.value
    return isinstance(cur, ast.Name) and cur.id in ("self", "cls")


def iter_names(node: ast.AST):
    """Yield every ``ast.Name`` anywhere under ``node``."""
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            yield n


def describe(node: ast.AST) -> str:
    """A human-readable label for a node in a violation message."""
    return dotted(node) or type(node).__name__


# ── whole-file scan ──────────────────────────────────────────────────────────────────────────
class FileScan(BaseModel):
    """Names harvested from the whole file, used to classify calls in the private region."""

    bindings: dict[
        str, str
    ]  # import alias -> fully-qualified module path (jnp -> jax.numpy)
    hidden_defs: set[str]  # class/func names defined inside the private region
    visible_defs: set[str]  # function names defined in the visible (public) region


def scan_file(tree: ast.Module, ranges) -> FileScan:
    """Collect import bindings and the class/func names defined inside vs. outside the private region.

    Imports live in the visible region (they're banned inside the private one), but their bindings are
    what the checker resolves private-region calls against — so we scan the whole file, not just the
    private lines.
    """
    bindings: dict[str, str] = {}
    hidden_defs: set[str] = set()
    visible_defs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bindings[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                bindings[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if node_in_ranges(node, ranges):
                hidden_defs.add(node.name)
            elif isinstance(node, ast.FunctionDef):
                visible_defs.add(node.name)
    return FileScan(
        bindings=bindings, hidden_defs=hidden_defs, visible_defs=visible_defs
    )
