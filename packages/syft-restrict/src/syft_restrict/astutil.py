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


def node_overlaps_ranges(node: ast.AST, ranges) -> bool:
    """True if a node's line SPAN intersects any range.

    Unlike ``node_in_ranges`` (which tests only the start line), this also catches a multi-line
    node that begins outside the ranges but extends into one -- a statement straddling the
    public/private boundary. Nodes with no position never overlap.
    """
    start = getattr(node, "lineno", None)
    if start is None:
        return False
    end = getattr(node, "end_lineno", None) or start
    return any(start <= hi and lo <= end for lo, hi in ranges)


# ── reading names and dotted paths off the tree ──────────────────────────────────────────────
def dotted_name(node: ast.AST) -> str | None:
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


def describe(node: ast.AST) -> str:
    """A human-readable label for a node in a violation message."""
    return dotted_name(node) or type(node).__name__


# ── whole-file scan ──────────────────────────────────────────────────────────────────────────
class FileScan(BaseModel):
    """Names harvested from the whole file, used to classify calls in the private region."""

    import_bindings: dict[
        str, str
    ]  # import alias -> fully-qualified module path (jnp -> jax.numpy)
    private_defs: set[str]  # class/func names defined inside the private region
    public_defs: set[str]  # class/func names defined in the public region (non-methods)
    private_def_ids: set[
        int
    ]  # id() of the first (canonical) node behind each private_defs name
    method_ids: set[int]  # id() of direct FunctionDef children of any ClassDef body


def scan_file(tree: ast.Module, private_ranges) -> FileScan:
    """Collect import bindings and the class/func names defined inside vs. outside the private region.

    Imports live in the public region (they're banned inside the private one), but their bindings are
    what the checker resolves private-region calls against — so we scan the whole file, not just the
    private lines.
    """
    import_bindings: dict[str, str] = {}  # import alias -> fully-qualified module path
    private_defs: set[str] = set()  # class/func names defined inside the private region
    public_defs: set[str] = set()  # class/func names defined in the public region
    private_def_ids: set[int] = (
        set()
    )  # id() of the first node claiming each private_defs name
    # Methods (direct FunctionDef children of a class body) are called via `self.<name>`, never
    # by bare name, so they're excluded here -- otherwise unrelated classes sharing a hook name
    # (`setup`, `__call__`) would look like one shadowing the other.
    method_ids = {
        id(child)
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        for child in node.body
        if isinstance(child, ast.FunctionDef)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # The alias (key) is the name the private code writes; the value is the
                # fully-qualified path the policy matches against. _Checker._resolve() swaps a
                # call's root name for this value and keeps the rest of the chain verbatim.
                # Bind the name Python actually binds at runtime:
                #   `import jax.numpy as jnp` binds `jnp` -> the jax.numpy module
                #   `import jax.numpy`        binds `jax` -> the jax PACKAGE (not jax.numpy!) --
                #      you reach save through `jax.numpy.save`, so the root resolves to itself.
                # Binding the root to the full dotted path would mis-resolve `jax.numpy.save` to
                # `jax.numpy.numpy.save` and let it slip a disallow floor (see tests/verify).
                if alias.asname:
                    import_bindings[alias.asname] = alias.name
                else:
                    root = alias.name.split(".")[0]
                    import_bindings[root] = root
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                import_bindings[alias.asname or alias.name] = (
                    f"{node.module}.{alias.name}"
                )
        elif isinstance(node, ast.FunctionDef) and id(node) in method_ids:
            continue
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if node_in_ranges(node, private_ranges):
                # `class Net:` / `def _secret():` on a private line -> private_defs.add("Net" / "_secret")
                # ast.walk is breadth-first, so the first node to claim a name is always the
                # shallowest (outermost) one -- any later def/class reusing the name is a shadow.
                if node.name not in private_defs:
                    private_def_ids.add(id(node))
                private_defs.add(node.name)
            else:
                # public FunctionDef *or* ClassDef — both may be called by bare name from private
                public_defs.add(node.name)
    return FileScan(
        import_bindings=import_bindings,
        private_defs=private_defs,
        public_defs=public_defs,
        private_def_ids=private_def_ids,
        method_ids=method_ids,
    )
