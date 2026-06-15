"""``run`` — orchestrate verify → obfuscate → certificate."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from .errors import PolicyViolation
from .obfuscator import obfuscate
from .policy import Policy
from .verifier import Violation, _normalize_ranges, _scan_file, verify

__all__ = ["run", "RunResult"]


@dataclass
class RunResult:
    ok: bool
    violations: list[Violation] = field(default_factory=list)
    obfuscated_path: str | None = None
    certificate: dict | None = None


def run(
    path: str | Path,
    private,
    allow_functions: str = "",
    allow_methods: str = "",
    out: str | Path | None = None,
    strict: bool = True,
) -> RunResult:
    """Verify the private region, then (on success) write an obfuscated copy.

    Args:
        path: the inference source file.
        private: list of ``[start, end]`` 1-based inclusive line ranges to hide + verify.
        allow_functions: comma-separated dotted-path globs callable by name (e.g. ``"jax.*, flax.linen.*"``).
        allow_methods: comma-separated operator bundles allowed on a value
            (``arithmetic, indexing, comparison, metadata``).
        out: where to write the obfuscated file (default ``<stem>.obfuscated.py`` next to the source).
        strict: if True (default), raise ``PolicyViolation`` when verification fails; otherwise return
            a ``RunResult`` with ``ok=False`` and no output written.
    """
    path = Path(path)
    source = path.read_text()
    policy = Policy.parse(allow_functions, allow_methods)

    result = verify(source, private, policy)
    if not result.ok:
        if strict:
            raise PolicyViolation(result.violations)
        return RunResult(ok=False, violations=result.violations)

    scan = _scan_file(ast.parse(source), _normalize_ranges(private))
    obfuscated = obfuscate(source, private, scan)

    out_path = Path(out) if out is not None else path.with_suffix(".obfuscated.py")
    out_path.write_text(obfuscated)

    certificate = {
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "policy_id": policy.policy_id(),
        "verifuscate_version": _version(),
        "private_ranges": [list(r) for r in _normalize_ranges(private)],
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
