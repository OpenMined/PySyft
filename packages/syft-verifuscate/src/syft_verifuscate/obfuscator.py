"""The display transform (research approach A).

``obfuscate(source, private, scan)`` returns a copy of ``source`` in which only the lines inside the
private ranges are transformed — identifiers renamed to neutral placeholders, constant values and
einsum-equation strings blanked to ``■``, and comments/docstrings stripped. Every line outside the
private ranges is emitted byte-for-byte, so the data owner can diff it against the original glue.

It is *display-only*: the obfuscated file is for reading, not running (the real, unobfuscated code is
what runs in the enclave). Renaming is deterministic — same input, same output.
"""

from __future__ import annotations

import ast
import io
import keyword
import tokenize

from .policy import DEFAULT_KEEP, METADATA_ATTRS
from .verifier import FileScan, _dotted, _is_dunder, _normalize_ranges

_BLANK = "■"  # ■

# Builtins kept readable (they reveal nothing about the architecture).
_KEEP_BUILTINS = frozenset(
    {
        "int",
        "float",
        "bool",
        "str",
        "bytes",
        "len",
        "range",
        "enumerate",
        "zip",
        "min",
        "max",
        "sum",
        "abs",
        "round",
        "all",
        "any",
        "tuple",
        "list",
        "dict",
        "set",
        "sorted",
        "reversed",
        "isinstance",
        "super",
        "None",
        "True",
        "False",
    }
)


def obfuscate(source: str, private, scan: FileScan) -> str:
    ranges = _normalize_ranges(private)
    tree = ast.parse(source)
    value_map, attr_map = _build_maps(tree, ranges, scan)

    keep_values = (
        DEFAULT_KEEP
        | set(scan.bindings)
        | set(scan.visible_defs)
        | _KEEP_BUILTINS
        | set(keyword.kwlist)
        | set(getattr(keyword, "softkwlist", []))
    )

    edits: list[tuple[int, int, int, int, str]] = []
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    prev_op_dot = False
    for tok in tokens:
        srow, scol = tok.start
        erow, ecol = tok.end
        if not _row_in_ranges(srow, ranges):
            if tok.type not in (
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
            ):
                prev_op_dot = tok.type == tokenize.OP and tok.string == "."
            continue

        if tok.type == tokenize.NAME:
            is_attr = prev_op_dot
            if is_attr:
                new = attr_map.get(tok.string)
            else:
                new = None if tok.string in keep_values else value_map.get(tok.string)
            if new is not None:
                edits.append((srow, scol, erow, ecol, new))
        elif tok.type == tokenize.STRING:
            edits.append((srow, scol, erow, ecol, f'"{_BLANK}"'))
        elif tok.type == tokenize.NUMBER:
            edits.append((srow, scol, erow, ecol, _BLANK))
        elif tok.type == tokenize.COMMENT:
            edits.append(
                (srow, scol, erow, ecol, "")
            )  # drop comments (incl. commented-out configs)

        if tok.type not in (
            tokenize.NL,
            tokenize.NEWLINE,
            tokenize.INDENT,
            tokenize.DEDENT,
        ):
            prev_op_dot = tok.type == tokenize.OP and tok.string == "."

    return _apply_edits(source, edits)


# ── build the deterministic rename maps from the AST ─────────────────────────────────────
def _build_maps(tree: ast.Module, ranges, scan: FileScan):
    keep_attrs: set[str] = set(METADATA_ATTRS)
    mangle_attr_names: set[str] = set()
    value_occurrences: list[tuple[tuple[int, int], str]] = []
    private_classes = _names_of(tree, ast.ClassDef, ranges)
    private_funcs = _names_of(tree, ast.FunctionDef, ranges)

    for node in ast.walk(tree):
        if not _node_in_ranges(node, ranges):
            continue
        if isinstance(node, ast.Attribute) and not _is_dunder(node.attr):
            root = (_dotted(node.value) or "").split(".")[0]
            if root in scan.bindings:
                keep_attrs.add(
                    node.attr
                )  # public library attr (e.g. jnp.einsum) — stays readable
            else:
                mangle_attr_names.add(node.attr)
        elif isinstance(node, ast.Name):
            value_occurrences.append(((node.lineno, node.col_offset), node.id))
        elif isinstance(node, ast.arg):
            value_occurrences.append(((node.lineno, node.col_offset), node.arg))

    # attr placeholders, in sorted name order for determinism
    attr_map: dict[str, str] = {}
    for i, name in enumerate(sorted(mangle_attr_names - keep_attrs)):
        attr_map[name] = f"░a{i}"

    # value placeholders, assigned in source order (first occurrence wins)
    keep_values = (
        DEFAULT_KEEP
        | set(scan.bindings)
        | set(scan.visible_defs)
        | _KEEP_BUILTINS
        | set(keyword.kwlist)
        | set(getattr(keyword, "softkwlist", []))
    )
    value_map: dict[str, str] = {}
    counters = {"cls": 0, "fn": 0, "v": 0}
    for _pos, name in sorted(value_occurrences):
        if name in keep_values or name in value_map:
            continue
        if name in private_classes:
            value_map[name] = f"░Cls{counters['cls']}"
            counters["cls"] += 1
        elif name in private_funcs:
            value_map[name] = f"░fn{counters['fn']}"
            counters["fn"] += 1
        else:
            value_map[name] = f"░v{counters['v']}"
            counters["v"] += 1
    return value_map, attr_map


def _names_of(tree, node_type, ranges) -> set[str]:
    return {
        n.name
        for n in ast.walk(tree)
        if isinstance(n, node_type) and _node_in_ranges(n, ranges)
    }


# ── apply position edits to the source, preserving non-private lines verbatim ─────────────
def _apply_edits(source: str, edits) -> str:
    lines = source.splitlines(keepends=True)
    # apply bottom-up so earlier edits don't shift later line indices / columns
    for srow, scol, erow, ecol, new in sorted(edits, reverse=True):
        if srow == erow:
            line = lines[srow - 1]
            lines[srow - 1] = line[:scol] + new + line[ecol:]
        else:
            merged = lines[srow - 1][:scol] + new + lines[erow - 1][ecol:]
            lines[srow - 1 : erow] = [merged]
    return "".join(lines)


def _row_in_ranges(row: int, ranges) -> bool:
    return any(lo <= row <= hi for lo, hi in ranges)


def _node_in_ranges(node: ast.AST, ranges) -> bool:
    line = getattr(node, "lineno", None)
    return line is not None and _row_in_ranges(line, ranges)
