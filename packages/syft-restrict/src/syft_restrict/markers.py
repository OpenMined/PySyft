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


def _scan_markers(source: str) -> dict[int, tuple[str, str | None]]:
    """Map each line number carrying a ``# syft-restrict: ...`` comment to its ``(kind, boundary)``.

    ``boundary`` is ``"start"``, ``"end"``, or ``None`` for a single-line marker. Only real
    comment tokens count, so a marker-shaped string inside a string literal is never mistaken
    for a directive.
    """
    markers: dict[int, tuple[str, str | None]] = {}
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type != tokenize.COMMENT:
            continue
        match = _MARKER_RE.match(tok.string.strip())
        if match:
            markers[tok.start[0]] = (match.group(1), match.group(2))
    return markers


def _compact(lines: list[int]) -> list[tuple[int, int]]:
    """Group a strictly increasing list of line numbers into maximal consecutive (lo, hi) runs."""
    ranges: list[tuple[int, int]] = []
    for line in lines:
        if ranges and ranges[-1][1] == line - 1:
            ranges[-1] = (ranges[-1][0], line)
        else:
            ranges.append((line, line))
    return ranges


class _MarkerParser:
    """Resolve ``# syft-restrict: ...`` comments into ``(obfuscate, hide)`` line ranges.

    A stack of open blocks (``(kind, start_line)``) tracks nesting. Every line in the file is
    attributed to whichever kind is on top of the stack (or to a single-line marker's own kind,
    if that line carries one); the accumulated per-kind line numbers are compacted into ranges
    at the end, instead of computing spans incrementally while scanning.

    Only ``hide`` may nest, and only one level deep inside an open ``obfuscate`` block -- hide is
    strictly stronger, so carving a stricter sub-region out of a looser one is safe. The reverse
    isn't allowed, and neither kind nests inside itself.
    """

    def __init__(self) -> None:
        self.stack: list[
            tuple[str, int]
        ] = []  # open blocks, innermost last: (kind, start_line)
        self._lines: dict[str, list[int]] = {"obfuscate": [], "hide": []}

    def feed(self, source: str) -> None:
        markers = _scan_markers(source)
        for line in range(1, len(source.splitlines()) + 1):
            marker = markers.get(line)
            if marker is None:
                if self.stack:
                    self._lines[self.stack[-1][0]].append(line)
                continue
            kind, boundary = marker
            if boundary == "start":
                self._push(kind, line)
            elif boundary == "end":
                self._pop(kind, line)
            else:
                self._single_line(kind, line)

    def resolve(self) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """Finalize the scan: raise on a still-open block or a total absence of markers."""
        if self.stack:
            okind, ostart = self.stack[
                -1
            ]  # innermost unterminated block, reported first
            raise MarkerError(
                f"line {ostart}: '{okind}-start' has no matching '{okind}-end'"
            )
        if not self._lines["obfuscate"] and not self._lines["hide"]:
            raise MarkerError(
                "no syft-restrict markers found in source; add `# syft-restrict: obfuscate-start` "
                "/ `# syft-restrict: obfuscate-end` (or `hide`) around the private code, or pass "
                "obfuscate=/hide= explicitly to run()"
            )
        return _compact(self._lines["obfuscate"]), _compact(self._lines["hide"])

    def _can_nest(self, kind: str) -> bool:
        return (
            len(self.stack) == 1 and self.stack[0][0] == "obfuscate" and kind == "hide"
        )

    def _nest_violation(
        self, kind: str, line: int, *, single_line: bool
    ) -> MarkerError:
        okind, ostart = self.stack[-1]
        reason = (
            "hide blocks cannot contain nested markers"
            if okind == "hide"
            else "only 'hide' may nest inside an open 'obfuscate' block"
        )
        shape = f"single-line '{kind}' marker" if single_line else f"'{kind}-start'"
        return MarkerError(
            f"line {line}: {shape} while '{okind}-start' is still open "
            f"(opened at line {ostart}); {reason}"
        )

    def _push(self, kind: str, line: int) -> None:
        if self.stack and not self._can_nest(kind):
            raise self._nest_violation(kind, line, single_line=False)
        self.stack.append((kind, line))

    def _pop(self, kind: str, line: int) -> None:
        if not self.stack:
            raise MarkerError(
                f"line {line}: '{kind}-end' has no matching '{kind}-start'"
            )
        okind, ostart = self.stack[-1]
        if okind != kind:
            hint = "; close the innermost block first" if len(self.stack) > 1 else ""
            raise MarkerError(
                f"line {line}: '{kind}-end' does not match the open '{okind}-start' block "
                f"(opened at line {ostart}){hint}"
            )
        if line <= ostart + 1:
            raise MarkerError(f"line {ostart}: '{okind}-start' block is empty")
        self.stack.pop()

    def _single_line(self, kind: str, line: int) -> None:
        if self.stack and not self._can_nest(kind):
            raise self._nest_violation(kind, line, single_line=True)
        self._lines[kind].append(line)


def parse_markers(source: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Scan ``source`` for ``# syft-restrict: ...`` markers and resolve them to line ranges.

    Returns ``(obfuscate_ranges, hide_ranges)``, both 1-based inclusive ``(lo, hi)`` tuples,
    excluding the marker comment lines themselves. Raises ``MarkerError`` on any unmatched,
    mismatched, badly-nested, or empty marker block, and when no marker is found at all.
    """
    parser = _MarkerParser()
    parser.feed(source)
    return parser.resolve()
