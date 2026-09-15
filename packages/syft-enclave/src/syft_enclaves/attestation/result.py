"""The shape every attestation verifier reports through.

One checklist type, so a Confidential Space token and a Tinfoil report read the
same way to whoever is appraising them. Lives apart from either verifier so
both can import it without the package importing itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


class AttestationError(Exception):
    """Raised when enclave attestation verification fails."""

    def __init__(self, message: str, result: AttestationResult | None = None):
        self.result = result
        super().__init__(message)


@dataclass
class CheckResult:
    name: str
    label: str
    passed: bool | None = None  # None = not yet run
    detail: str = ""


@dataclass
class AttestationResult:
    checks: list[CheckResult] = field(default_factory=list)
    #: The peer's syft public key bundle, when it arrived over a channel bound
    #: to the attestation report (see ``attestation.https``). None whenever
    #: there was no such channel — a bundle read from Drive is not bound to
    #: anything and must not be set here.
    verified_key_bundle: Optional[dict] = None

    def add(self, name: str, label: str, passed: bool, detail: str) -> None:
        self.checks.append(
            CheckResult(name=name, label=label, passed=passed, detail=detail)
        )

    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def first_failure(self) -> CheckResult | None:
        return next((c for c in self.checks if not c.passed), None)

    def print_checklist(self) -> None:
        for check in self.checks:
            if check.passed is None:
                icon = "  ⏭️"
            elif check.passed:
                icon = "  ✅"
            else:
                icon = "  ❌"
            print(f"{icon} {check.label:<20s} — {check.detail}")
