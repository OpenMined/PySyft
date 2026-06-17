"""``run`` — orchestrate verify → obfuscate → certificate."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from pydantic import BaseModel, Field

from .errors import PolicyViolation
from .obfuscator import (
    obfuscate as _obfuscate,
)  # aliased: `obfuscate` is also a run() kwarg
from .policy import Policy
from .verifier import Violation, _normalize_ranges, _scan_file, verify

__all__ = ["run", "RunResult"]


class RunResult(BaseModel):
    ok: bool
    violations: list[Violation] = Field(default_factory=list)
    obfuscated_path: str | None = None
    certificate: dict | None = None


def run(
    path: str | Path,
    obfuscate=None,
    hide=None,
    allow_functions: list[str] | None = None,
    allow_methods: list[str] | None = None,
    out: str | Path | None = None,
    strict: bool = True,
) -> RunResult:
    """Verify the private region, then (on success) write a display copy.

    The private region is the *union* of ``obfuscate`` and ``hide`` — both are secret code that runs
    in the enclave, so both are verified. They differ only in how the display copy renders them:

    Args:
        path: the inference source file.
        obfuscate: ``[start, end]`` 1-based inclusive line ranges to *obfuscate* (identifiers renamed,
            constants blanked, structure preserved).
        hide: ``[start, end]`` 1-based inclusive line ranges to *hide* (whole line replaced with a
            ``■■■■■■■■`` marker, indentation kept).
        allow_functions: list of dotted-path globs callable by name (e.g. ``["jax.*", "flax.linen.*"]``).
        allow_methods: list of operator bundles allowed on a value
            (``["arithmetic", "indexing", "comparison"]``).
        out: where to write the obfuscated file (default ``<stem>.obfuscated.py`` next to the source).
        strict: if True (default), raise ``PolicyViolation`` when verification fails; otherwise return
            a ``RunResult`` with ``ok=False`` and no output written.
    """
    path = Path(path)
    source = path.read_text()
    policy = Policy.parse(allow_functions, allow_methods)

    obfuscate_ranges = obfuscate or []
    hide_ranges = hide or []
    private = [*obfuscate_ranges, *hide_ranges]  # union = the verified region

    result = verify(source, private, policy)
    if not result.ok:
        if strict:
            raise PolicyViolation(result.violations)
        return RunResult(ok=False, violations=result.violations)

    scan = _scan_file(ast.parse(source), _normalize_ranges(private))
    obfuscated = _obfuscate(source, obfuscate_ranges, hide_ranges, scan)

    out_path = Path(out) if out is not None else path.with_suffix(".obfuscated.py")
    out_path.write_text(obfuscated)

    certificate = {
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "policy_id": policy.policy_id(),
        "verifuscate_version": _version(),
        "private_ranges": [list(r) for r in _normalize_ranges(private)],
        "obfuscate_ranges": [list(r) for r in _normalize_ranges(obfuscate_ranges)],
        "hide_ranges": [list(r) for r in _normalize_ranges(hide_ranges)],
        "n_calls_checked": result.n_calls_checked,
    }
    return RunResult(
        ok=True,
        violations=[],
        obfuscated_path=str(out_path),
        certificate=certificate,
    )


def _version() -> str:
    from . import __version__

    return __version__
