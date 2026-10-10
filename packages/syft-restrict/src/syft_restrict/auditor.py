"""Advisory audit of an ``allow_functions`` list: classify each dotted path as ``unsafe`` /
``dual_use`` / ``safe`` / ``review`` against a catalog passed via ``catalog_dir``.

Advisory, not a proof — the verifier's default-deny is what gates calls. Classification is catalog
matching only (no source inspection); an uncatalogued path is ``review``, never assumed safe. No
catalog ships with the package. See docs/audit.md for the verdicts, catalog layout, and workflow.
"""

from __future__ import annotations

import fnmatch
import importlib.metadata
import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

__all__ = ["audit_allow_functions", "AuditReport", "PathAudit"]

# A catalog root is a directory laid out as <library>/<version>/catalog.json, plus _common/default
# for library-agnostic rules. None ships with the package; callers pass one via ``catalog_dir``.
# See docs/audit.md for the scheme and examples/catalog for a worked example.
_COMMON_LIB = "_common"
_COMMON_VERSION = "default"

Verdict = Literal["safe", "dual_use", "unsafe", "review"]

# Catalog buckets, in the order they are matched (first hit wins): the strictest verdict a path
# qualifies for is assigned, so unsafe beats dual_use beats safe.
_BUCKETS: tuple[Verdict, ...] = ("unsafe", "dual_use", "safe")


class PathAudit(BaseModel):
    path: str
    verdict: Verdict
    reason: str


class AuditReport(BaseModel):
    entries: list[PathAudit] = Field(default_factory=list)
    versions: dict[str, str] = Field(
        default_factory=dict
    )  # top-level package -> version seen

    def _by_verdict(self, verdict: Verdict) -> list[PathAudit]:
        return [e for e in self.entries if e.verdict == verdict]

    @property
    def unsafe(self) -> list[PathAudit]:
        return self._by_verdict("unsafe")

    @property
    def dual_use(self) -> list[PathAudit]:
        return self._by_verdict("dual_use")

    @property
    def review(self) -> list[PathAudit]:
        return self._by_verdict("review")

    @property
    def safe(self) -> list[PathAudit]:
        return self._by_verdict("safe")

    @property
    def ok(self) -> bool:
        """True if nothing is unsafe. ``dual_use`` and ``review`` entries do not fail it -- they are
        allowed-but-flagged and need a human's eye, not a hard block."""
        return not self.unsafe

    def format(self) -> str:
        vers = (
            ", ".join(f"{k} {v}" for k, v in sorted(self.versions.items()))
            or "no versions detected"
        )
        lines = [f"allow-list audit ({vers})"]
        groups = (
            ("UNSAFE", self.unsafe),
            ("DUAL-USE", self.dual_use),
            ("REVIEW", self.review),
            ("SAFE", self.safe),
        )
        for label, group in groups:
            if not group:
                continue
            lines.append(f"  {label} ({len(group)}):")
            for e in group:
                lines.append(f"    - {e.path}{' — ' + e.reason if e.reason else ''}")
        lines.append(
            f"  => ok={self.ok} (unsafe entries fail; dual-use and review entries need a human)"
        )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format()


def audit_allow_functions(
    allow_functions: list[str] | None, *, catalog_dir: str | Path | None = None
) -> AuditReport:
    """Classify each entry of an ``allow_functions`` list; see the module docstring for semantics.

    Classification is catalog matching only: a path the catalog does not know is reported as
    ``"review"`` and deferred to a human. Unknowns are never assumed safe.

    ``catalog_dir`` is the external catalog root (``<library>/<version>/catalog.json`` layout). No
    catalog ships with the package, so without ``catalog_dir`` there are no rules and every path is
    reported as ``"review"``. A worked example lives in ``examples/catalog``.
    """
    paths = [p for p in (s.strip() for s in (allow_functions or [])) if p]
    versions = _detect_versions(paths)
    root = Path(catalog_dir) if catalog_dir is not None else None
    entries = [_classify(p, versions, root) for p in paths]
    return AuditReport(entries=entries, versions=versions)


def _classify(path: str, versions: dict[str, str], root: Path | None) -> PathAudit:
    if "*" in path or "?" in path:
        return PathAudit(
            path=path,
            verdict="unsafe",
            reason="glob allow grants an entire namespace (may include disk/network/callbacks); "
            "list exact leaves instead",
        )
    library = path.split(".", 1)[0]
    rules = _rules_for(library, versions.get(library, ""), root)
    for verdict in _BUCKETS:  # unsafe -> dual_use -> safe: strictest match wins
        for pattern, reason in rules[verdict].items():
            if fnmatch.fnmatchcase(path, pattern):
                return PathAudit(path=path, verdict=verdict, reason=reason)
    return PathAudit(
        path=path,
        verdict="review",
        reason="not in the curated catalog; defer to human review (unknowns are not assumed safe)",
    )


def _rules_for(
    library: str, version: str, root: Path | None
) -> dict[str, dict[str, str]]:
    """Merge the library-agnostic ``_common`` rules with the version-matched library rules from the
    catalog ``root``, per bucket (``unsafe`` / ``dual_use`` / ``safe``). No ``root`` -> no rules."""
    merged: dict[str, dict[str, str]] = {bucket: {} for bucket in _BUCKETS}
    if root is None:
        return merged
    common = _load_ruleset(root, _COMMON_LIB, _COMMON_VERSION)
    version_dir = _match_version_dir(root, library, version)
    lib_rules = (
        _load_ruleset(root, library, version_dir) if version_dir is not None else {}
    )
    for ruleset in (common, lib_rules):
        for bucket in _BUCKETS:
            merged[bucket].update(ruleset.get(bucket, {}))
    return merged


def _load_ruleset(root: Path, library: str, version_dir: str) -> dict:
    """Read ``<root>/<library>/<version_dir>/catalog.json``; empty dict if it is absent, unreadable,
    or malformed. A broken catalog file yields no rules (its paths fall to ``review``) rather than
    crashing the advisory audit."""
    path = root / library / version_dir / "catalog.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _match_version_dir(root: Path, library: str, version: str) -> str | None:
    """Longest version-dir under ``<root>/<library>/`` matching ``version`` on a dot boundary.

    Returns ``None`` when no directory matches — there is no version-agnostic fallback, so an
    uncovered version simply contributes no library rules.
    """
    lib_dir = root / library
    if not lib_dir.is_dir():
        return None
    keys = [child.name for child in lib_dir.iterdir() if child.is_dir()]
    return _best_version_key(keys, version)


def _best_version_key(keys: list[str], version: str) -> str | None:
    """Longest key that equals ``version`` or is a dot-bounded prefix of it: "0.1" covers 0.1.x,
    never 0.11.x / 0.19.x. Returns ``None`` if nothing matches (no baseline fallback)."""
    best, best_len = None, -1
    if not version:
        return None
    for key in keys:
        if (version == key or version.startswith(key + ".")) and len(key) > best_len:
            best, best_len = key, len(key)
    return best


def _detect_versions(paths) -> dict[str, str]:
    """Resolve the installed version of each top-level package named in ``paths`` **without importing
    it** (no import side effects): locate it with ``find_spec`` and read distribution metadata."""
    dist_map = (
        importlib.metadata.packages_distributions()
    )  # import name -> [distribution names]
    versions: dict[str, str] = {}
    for path in paths:
        root = path.split(".", 1)[0].lstrip("*")
        if not root or root in versions:
            continue
        try:
            spec = importlib.util.find_spec(root)
        except (ImportError, ValueError):
            spec = None
        versions[root] = (
            _dist_version(root, dist_map) if spec is not None else "not installed"
        )
    return versions


def _dist_version(root: str, dist_map: Mapping[str, list[str]]) -> str:
    """Version of the distribution providing top-level import ``root``, or ``"unknown"`` if it has no
    resolvable distribution metadata (e.g. a namespace package or a local module)."""
    candidates = [
        *dist_map.get(root, []),
        root,
    ]  # metadata name may differ from the import name
    for dist in candidates:
        try:
            return importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"
