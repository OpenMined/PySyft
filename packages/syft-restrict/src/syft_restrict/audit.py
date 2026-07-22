"""Advisory audit of an ``allow_functions`` list: classify each allowed dotted path by risk.

This is **defense-in-depth, not a soundness proof.** The verifier's default-deny already decides what
the private region may call; this tool helps an author or reviewer see — *before* running — whether
their allow-list grants any known disk/network/host-callback capability, or any path a human should
eyeball. It generalizes across models: feed it whatever ``allow_functions`` a given model needs.

Each allowed entry is classified against a **curated catalog** (the ``catalog/`` directory, laid out
as ``catalog/<library>/<version>/catalog.json`` plus ``catalog/_common/default`` for library-agnostic
rules — kept out of the code so assessments can be revised per release; see ``catalog/README.md``):

- ``"unsafe"`` — matches a catalog entry for known disk/network/host-callback surface, OR is a glob
  (``jax.*``) that grants a whole namespace. Remove it or tighten the allow.
- ``"dual_use"`` — a useful, mostly-safe op that can still be abused in combination (e.g. ``einsum``,
  ``softmax``, ``where``). Allowed, but flagged: the *category itself* carries the "handle with care"
  signal, so entry notes stay terse and vague — never an abuse how-to.
- ``"safe"`` — matches a curated entry for a genuinely inert path (constants, masks, module refs) with
  no residual output channel of its own.
- ``"review"`` — none of the above. The audit makes **no** guess about it: it is reported as
  uncatalogued and deferred to human review. Unknowns are never assumed safe.

Limits (state them plainly to whoever reads a report):

- Classification is **only** catalog matching — no source inspection, no inference. A path the catalog
  does not know is deferred to a human; the tool does not try to guess whether it does I/O.
- The catalog is curated per library version; the report records the versions it saw.

Anything not matched as unsafe, dual_use, or safe defaults to ``"review"``, never silently to safe.
"""

from __future__ import annotations

import fnmatch
import importlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

__all__ = ["audit_allow_functions", "AuditReport", "PathAudit"]

# Catalog laid out as catalog/<library>/<version>/catalog.json, plus catalog/_common/default for
# library-agnostic rules. See catalog/README.md for the scheme.
_CATALOG_DIR = Path(__file__).with_name("catalog")
_COMMON_LIB = "_common"
_COMMON_VERSION = "default"

# Catalog buckets, in the order they are matched (first hit wins): the strictest verdict a path
# qualifies for is assigned, so unsafe beats dual_use beats safe.
_BUCKETS: tuple[str, ...] = ("unsafe", "dual_use", "safe")

Verdict = Literal["safe", "dual_use", "unsafe", "review"]


class PathAudit(BaseModel):
    path: str
    verdict: Verdict
    reason: str


class AuditReport(BaseModel):
    entries: list[PathAudit] = Field(default_factory=list)
    versions: dict[str, str] = Field(default_factory=dict)  # top-level package -> version seen

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
        vers = ", ".join(f"{k} {v}" for k, v in sorted(self.versions.items())) or "no versions detected"
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
        lines.append(f"  => ok={self.ok} (unsafe entries fail; dual-use and review entries need a human)")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format()


def audit_allow_functions(allow_functions: list[str] | None) -> AuditReport:
    """Classify each entry of an ``allow_functions`` list; see the module docstring for semantics.

    Classification is catalog matching only: a path the catalog does not know is reported as
    ``"review"`` and deferred to a human. Unknowns are never assumed safe.
    """
    paths = [p for p in (s.strip() for s in (allow_functions or [])) if p]
    versions = _detect_versions(paths)
    entries = [_classify(p, versions) for p in paths]
    return AuditReport(entries=entries, versions=versions)


def _classify(path: str, versions: dict[str, str]) -> PathAudit:
    if "*" in path or "?" in path:
        return PathAudit(
            path=path,
            verdict="unsafe",
            reason="glob allow grants an entire namespace (may include disk/network/callbacks); "
            "list exact leaves instead",
        )
    library = path.split(".", 1)[0]
    rules = _rules_for(library, versions.get(library, ""))
    for verdict in _BUCKETS:  # unsafe -> dual_use -> safe: strictest match wins
        for pattern, reason in rules[verdict].items():
            if fnmatch.fnmatchcase(path, pattern):
                return PathAudit(path=path, verdict=verdict, reason=reason)
    return PathAudit(
        path=path,
        verdict="review",
        reason="not in the curated catalog; defer to human review (unknowns are not assumed safe)",
    )


def _rules_for(library: str, version: str) -> dict[str, dict[str, str]]:
    """Merge the library-agnostic ``_common`` rules with the version-matched library rules, per
    bucket (``unsafe`` / ``dual_use`` / ``safe``)."""
    merged: dict[str, dict[str, str]] = {bucket: {} for bucket in _BUCKETS}
    common = _load_ruleset(_COMMON_LIB, _COMMON_VERSION)
    version_dir = _match_version_dir(library, version)
    lib_rules = _load_ruleset(library, version_dir) if version_dir is not None else {}
    for ruleset in (common, lib_rules):
        for bucket in _BUCKETS:
            merged[bucket].update(ruleset.get(bucket, {}))
    return merged


@lru_cache(maxsize=None)
def _load_ruleset(library: str, version_dir: str) -> dict:
    """Read ``catalog/<library>/<version_dir>/catalog.json``; empty dict if the file is absent."""
    path = _CATALOG_DIR / library / version_dir / "catalog.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def _match_version_dir(library: str, version: str) -> str | None:
    """Longest version-dir under ``catalog/<library>/`` matching ``version`` on a dot boundary.

    Returns ``None`` when no directory matches — there is no version-agnostic fallback, so an
    uncovered version simply contributes no library rules.
    """
    lib_dir = _CATALOG_DIR / library
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
    versions: dict[str, str] = {}
    for path in paths:
        root = path.split(".", 1)[0].lstrip("*")
        if not root or root in versions:
            continue
        try:
            mod = importlib.import_module(root)
        except ImportError:
            versions[root] = "not installed"
        else:
            versions[root] = getattr(mod, "__version__", "unknown")
    return versions
