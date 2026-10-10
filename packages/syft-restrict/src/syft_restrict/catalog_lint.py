#!/usr/bin/env python3
"""Lint the ``catalog.json`` files a benchmark owner writes for the allow-list audit.

Enforces a canonical, diff-friendly form:

- keys sorted at every mapping level (so reordering entries never shows up as a diff),
- 2-space indent, UTF-8 kept verbatim (no ``\\uXXXX`` escaping), trailing newline,
- no line breaks inside any string value (each entry description stays on one line),
- no entry a stricter bucket already matches (see ``_overlaps``).

Usage (installed as the ``syft-restrict-lint`` console script)::

    uv run syft-restrict-lint PATH          # check only; non-zero exit if anything is off
    uv run syft-restrict-lint PATH --fix     # rewrite files into canonical form in place

``PATH`` is a ``catalog.json`` file or a directory to lint recursively. ``--fix`` reformats (sorts +
indents) and collapses any line break inside a value into a single space; overlaps it only reports,
since resolving one is a judgement about the entry.
"""

from __future__ import annotations

import argparse
import difflib
import fnmatch
import json
import re
import sys
from pathlib import Path

# A run of whitespace containing at least one line break -> a single space.
_LINE_BREAK = re.compile(r"\s*[\r\n]+\s*")

# Buckets strictest first, mirroring the order the auditor matches them in.
_BUCKETS = ("unsafe", "dual_use", "safe")


def _canonical(data: object) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def _normalize(data: object) -> object:
    """Recursively collapse line breaks in every string value to a single space."""
    if isinstance(data, dict):
        return {key: _normalize(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_normalize(value) for value in data]
    if isinstance(data, str):
        return _LINE_BREAK.sub(" ", data)
    return data


def _overlaps(data: object) -> list[str]:
    """Report entries a stricter bucket already matches.

    The auditor takes the first hit in ``unsafe`` -> ``dual_use`` -> ``safe`` order, so a weaker
    entry the stricter bucket also matches never applies. Nothing fails and the audit still returns
    the stricter verdict, which makes the weaker entry a claim the catalog never honours.
    """
    if not isinstance(data, dict):
        return []
    # strictest first, keys sorted so the report is stable
    buckets: list[tuple[str, list[str]]] = []
    for name in _BUCKETS:
        entries = data.get(name)
        if isinstance(entries, dict) and entries:
            buckets.append((name, sorted(str(key) for key in entries)))

    problems: list[str] = []
    for index, (weaker, keys) in enumerate(buckets):
        for key in keys:
            for stricter, patterns in buckets[:index]:
                hit = next((p for p in patterns if fnmatch.fnmatchcase(key, p)), None)
                if hit is not None:
                    problems.append(
                        f"{key!r} in {weaker!r} is already matched by {hit!r} in {stricter!r}"
                    )
                    break  # one stricter match is enough to make the entry dead
    return problems


def _lint_file(path: Path, root: Path, *, fix: bool) -> list[str]:
    """Return a list of human-readable problems with ``path`` (empty if clean)."""
    rel = path.relative_to(root)
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return [f"{rel}: invalid JSON ({exc})"]

    # Reported in both modes: --fix cannot choose which bucket an entry belongs in.
    problems = [f"{rel}: {overlap}" for overlap in _overlaps(data)]

    canonical = _canonical(_normalize(data))
    if text == canonical:
        return problems

    if fix:
        path.write_text(canonical, encoding="utf-8")
        return problems

    diff = "".join(
        difflib.unified_diff(
            text.splitlines(keepends=True),
            canonical.splitlines(keepends=True),
            fromfile=f"{rel} (current)",
            tofile=f"{rel} (canonical)",
        )
    )
    problems.append(f"{rel}: not in canonical form (run --fix)\n{diff}".rstrip())
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        help="a catalog.json file, or a directory of catalog.json files to lint recursively",
    )
    parser.add_argument(
        "--fix", action="store_true", help="rewrite files into canonical form"
    )
    args = parser.parse_args(argv)

    target = Path(args.path)
    if not target.exists():
        print(f"path not found: {target}", file=sys.stderr)
        return 1
    if target.is_file():
        root, files = target.parent, [target]
    else:
        root, files = target, sorted(target.rglob("catalog.json"))
    if not files:
        print(f"no catalog.json found under {target}", file=sys.stderr)
        return 1

    problems: list[str] = []
    for path in files:
        problems += _lint_file(path, root, fix=args.fix)

    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1  # --fix clears formatting problems, but never an overlap

    print(f"catalog lint: {len(files)} file(s) OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
