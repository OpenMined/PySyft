"""Exceptions raised by syft-verifuscate."""

from __future__ import annotations


class VerifuscateError(Exception):
    """Base class for all verifuscate errors."""


class PolicyViolation(VerifuscateError):
    """Raised by ``run(..., strict=True)`` when the private region fails verification.

    The offending findings are attached as ``.violations`` (a list of ``Violation``).
    """

    def __init__(self, violations):
        self.violations = list(violations)
        lines = "\n".join(
            f"  line {v.line}: [{v.code}] {v.message}" for v in self.violations
        )
        super().__init__(
            f"verifuscate refused: {len(self.violations)} policy violation(s) in the private region:\n"
            f"{lines}"
        )
