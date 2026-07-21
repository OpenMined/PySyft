"""``run`` — orchestrate verify → obfuscate → certificate."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from pydantic import BaseModel, Field

from .astutil import normalize_ranges, scan_file
from .errors import PolicyViolation
from .markers import parse_markers
from .obfuscator import obfuscate as _obfuscate
from .policy import Policy
from .verifier import Violation, verify

__all__ = ["run", "RunResult"]


class RunResult(BaseModel):
    ok: bool
    violations: list[Violation] = Field(default_factory=list)
    obfuscated_path: str | None = None
    certificate: dict | None = None


def run(
    path: str | Path,
    allow_functions: list[str] | None = None,
    allow_operators: list[str] | None = None,
    disallow_functions: list[str] | None = None,
    allow_local_assignments: bool = True,
    allow_base_class_attributes: bool = True,
    out: str | Path | None = None,
    strict: bool = True,
) -> RunResult:
    """Verify the private region, then (on success) write a display copy.

    The private region is marked in the source with ``# syft-restrict: ...`` comments (see
    ``markers.parse_markers`` and docs/verify.md); this is the only supported way to designate it.
    Its ``obfuscate`` and ``hide`` sub-regions are resolved from those markers and both verified —
    they differ only in how the display copy renders them (identifiers renamed vs. whole lines
    blanked). A file with no markers raises ``MarkerError``.

    Args:
        path: the inference source file (must carry ``# syft-restrict: ...`` markers).
        allow_functions: list of dotted-path globs callable by name (e.g. ``["jax.*", "flax.linen.*"]``).
        allow_operators: list of operator bundles allowed on a value
            (``["arithmetic", "indexing", "comparison"]``).
        disallow_functions: optional list of dotted-path globs that BEAT the allow (e.g.
            ``["jax.numpy.save", "jax.experimental.*"]``). A hard floor for authors who allow a
            broad glob; empty by default, in which case only ``allow_functions`` applies.
        allow_local_assignments: if True (default), a local aliased to a safe callable may itself be
            called by name; if False, callables must be called directly.
        allow_base_class_attributes: if True (default), a ``self.<attr>`` never assigned in the class
            is presumed inherited from the (vetted) base and callable; if False, only assigned attrs are.
        out: where to write the obfuscated file (default ``<stem>.obfuscated.py`` next to the source).
        strict: if True (default), raise ``PolicyViolation`` when verification fails; otherwise return
            a ``RunResult`` with ``ok=False`` and no output written.
    """
    path = Path(path)
    # Read the source exactly once and hand it to _run, so marker resolution, verification, and
    # obfuscation all operate on identical bytes (no second read that could race / TOCTOU).
    source = path.read_text()
    obfuscate_ranges, hide_ranges = parse_markers(source)
    return _run(
        path,
        obfuscate=obfuscate_ranges,
        hide=hide_ranges,
        allow_functions=allow_functions,
        allow_operators=allow_operators,
        disallow_functions=disallow_functions,
        allow_local_assignments=allow_local_assignments,
        allow_base_class_attributes=allow_base_class_attributes,
        out=out,
        strict=strict,
        source=source,
    )


def _run(
    path: str | Path,
    obfuscate=None,
    hide=None,
    allow_functions: list[str] | None = None,
    allow_operators: list[str] | None = None,
    disallow_functions: list[str] | None = None,
    allow_local_assignments: bool = True,
    allow_base_class_attributes: bool = True,
    out: str | Path | None = None,
    strict: bool = True,
    source: str | None = None,
) -> RunResult:
    """Verify and obfuscate using explicit 1-based line ranges (no marker scanning).

    Internal entry point behind ``run()``. Callers that want to bypass the comment-marker UX and
    supply their own ranges may use it directly, accepting responsibility for those ranges.

    Args mirror ``run()`` except the private region is given as explicit ranges:
        obfuscate: ``[start, end]`` 1-based inclusive line ranges to *obfuscate* (identifiers renamed,
            constants blanked, structure preserved).
        hide: ``[start, end]`` 1-based inclusive line ranges to *hide* (whole line replaced with a
            ``■■■■■■■■`` marker, indentation kept). The verified region is the union of the two.
        source: the already-read file contents. When ``run()`` calls this, it passes the exact bytes
            it resolved markers against so nothing is read twice; otherwise the file is read here.
    """
    path = Path(path)
    if source is None:
        source = path.read_text()
    policy = Policy.parse(
        allow_functions,
        allow_operators,
        disallow_functions,
        allow_local_assignments,
        allow_base_class_attributes,
    )

    obfuscate_ranges = obfuscate or []
    hide_ranges = hide or []
    private = [*obfuscate_ranges, *hide_ranges]  # union = the verified region

    result = verify(source, private, policy)
    if not result.ok:
        if strict:
            raise PolicyViolation(result.violations)
        return RunResult(ok=False, violations=result.violations)

    scan = scan_file(ast.parse(source), normalize_ranges(private))
    obfuscated = _obfuscate(source, obfuscate_ranges, hide_ranges, scan)

    out_path = Path(out) if out is not None else path.with_suffix(".obfuscated.py")
    out_path.write_text(obfuscated)

    certificate = {
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "policy_id": policy.policy_id(),
        "restrict_version": _version(),
        "private_ranges": [list(r) for r in normalize_ranges(private)],
        "obfuscate_ranges": [list(r) for r in normalize_ranges(obfuscate_ranges)],
        "hide_ranges": [list(r) for r in normalize_ranges(hide_ranges)],
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
