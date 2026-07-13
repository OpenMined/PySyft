"""Comment-based markup, an alternative to hand-counted line ranges.

Instead of passing ``obfuscate``/``hide`` as ``[start, end]`` line numbers to ``run()``, a source
file may mark its own private/hidden regions with comments::

    # syft-restrict: obfuscate-start
    def attention(x):
        ...
    # syft-restrict: obfuscate-end

    MODEL_ID = "gemma-2b"  # syft-restrict: hide

A ``hide`` block (or single-line marker) may nest inside an open ``obfuscate`` block: hide is a
strictly stronger transform (whole line blanked) than obfuscate (structure preserved, identifiers
renamed), so carving out a stricter sub-region is safe. The reverse is not: obfuscate cannot nest
inside hide, and neither kind may nest inside itself.

``parse_markers(source)`` resolves these into the same ``(obfuscate_ranges, hide_ranges)`` shape
``run()`` already accepts. ``run()`` uses this automatically when both ``obfuscate`` and ``hide``
are omitted.
"""

from __future__ import annotations

import io
import re
import tokenize

from .errors import MarkerError

__all__ = ["parse_markers"]

# Matches the whole comment, not a substring, so a marker string that happens to appear inside an
# ordinary sentence-like comment is never mistaken for a directive.
_MARKER_RE = re.compile(r"^#\s*syft-restrict:\s*(obfuscate|hide)(?:-(start|end))?\s*$")


def parse_markers(source: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Scan ``source`` for ``# syft-restrict: ...`` markers and resolve them to line ranges.

    Returns ``(obfuscate_ranges, hide_ranges)``, both 1-based inclusive ``(lo, hi)`` tuples,
    excluding the marker comment lines themselves. Raises ``MarkerError`` on any unmatched,
    mismatched, badly-nested, or empty marker block, and when no marker is found at all.
    """
    ranges: dict[str, list[tuple[int, int]]] = {"obfuscate": [], "hide": []}
    outer: tuple[str, int] | None = (
        None  # (kind, start_line) of the open top-level block
    )
    inner: tuple[str, int] | None = (
        None  # (kind, start_line) of a hide block nested in outer
    )
    pending_start = 0  # next unclaimed line of the outer block, once one is open

    def close_outer_span(upto: int) -> None:
        """Claim the outer block's own lines up to (and including) line ``upto`` for its kind."""
        assert outer is not None
        okind, _ = outer
        if pending_start <= upto:
            ranges[okind].append((pending_start, upto))

    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type != tokenize.COMMENT:
            continue
        match = _MARKER_RE.match(tok.string.strip())
        if not match:
            continue
        kind, boundary = match.group(1), match.group(2)
        line = tok.start[0]

        if boundary is None:
            if inner is not None:
                ikind, istart = inner
                raise MarkerError(
                    f"line {line}: single-line '{kind}' marker inside an open "
                    f"'{ikind}-start' block (opened at line {istart}); hide blocks cannot "
                    "contain nested markers"
                )
            if outer is not None:
                okind, ostart = outer
                if okind == "obfuscate" and kind == "hide":
                    close_outer_span(line - 1)
                    ranges["hide"].append((line, line))
                    pending_start = line + 1
                else:
                    raise MarkerError(
                        f"line {line}: single-line '{kind}' marker inside an open "
                        f"'{okind}-start' block (opened at line {ostart})"
                    )
            else:
                ranges[kind].append((line, line))

        elif boundary == "start":
            if inner is not None:
                ikind, istart = inner
                raise MarkerError(
                    f"line {line}: '{kind}-start' while '{ikind}-start' is still open "
                    f"(opened at line {istart}); hide blocks cannot contain nested markers"
                )
            if outer is not None:
                okind, ostart = outer
                if okind == "obfuscate" and kind == "hide":
                    close_outer_span(line - 1)
                    inner = (kind, line)
                else:
                    raise MarkerError(
                        f"line {line}: '{kind}-start' while '{okind}-start' is still open "
                        f"(opened at line {ostart}); only 'hide' may nest inside an open "
                        "'obfuscate' block"
                    )
            else:
                outer = (kind, line)
                pending_start = line + 1

        else:  # boundary == "end"
            if inner is not None:
                ikind, istart = inner
                if ikind != kind:
                    raise MarkerError(
                        f"line {line}: '{kind}-end' does not match the open "
                        f"'{ikind}-start' block (opened at line {istart}); close the "
                        "innermost block first"
                    )
                if line <= istart + 1:
                    raise MarkerError(f"line {istart}: '{ikind}-start' block is empty")
                ranges["hide"].append((istart + 1, line - 1))
                pending_start = line + 1
                inner = None
            elif outer is not None:
                okind, ostart = outer
                if okind != kind:
                    raise MarkerError(
                        f"line {line}: '{kind}-end' does not match the open "
                        f"'{okind}-start' block (opened at line {ostart})"
                    )
                if line <= ostart + 1:
                    raise MarkerError(f"line {ostart}: '{okind}-start' block is empty")
                close_outer_span(line - 1)
                outer = None
            else:
                raise MarkerError(
                    f"line {line}: '{kind}-end' has no matching '{kind}-start'"
                )

    if inner is not None:
        ikind, istart = inner
        raise MarkerError(
            f"line {istart}: '{ikind}-start' has no matching '{ikind}-end'"
        )
    if outer is not None:
        okind, ostart = outer
        raise MarkerError(
            f"line {ostart}: '{okind}-start' has no matching '{okind}-end'"
        )

    if not ranges["obfuscate"] and not ranges["hide"]:
        raise MarkerError(
            "no syft-restrict markers found in source; add `# syft-restrict: obfuscate-start` / "
            "`# syft-restrict: obfuscate-end` (or `hide`) around the private code, or pass "
            "obfuscate=/hide= explicitly to run()"
        )

    return ranges["obfuscate"], ranges["hide"]
