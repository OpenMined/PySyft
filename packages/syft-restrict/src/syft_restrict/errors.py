"""Exceptions raised by syft-restrict."""

from __future__ import annotations


class RestrictError(Exception):
    """Base class for all restrict errors."""


class PolicyViolation(RestrictError):
    """Raised by ``run(..., strict=True)`` when the private region fails verification.

    The offending findings are attached as ``.violations`` (a list of ``Violation``).
    """

    def __init__(self, violations):
        self.violations = list(violations)
        lines = "\n".join(
            f"  line {v.line}: [{v.code}] {v.message}" for v in self.violations
        )
        super().__init__(
            f"restrict refused: {len(self.violations)} policy violation(s) in the private region:\n"
            f"{lines}"
        )


class MarkerError(RestrictError):
    """Raised when ``# syft-restrict: ...`` markers in a source file are malformed or absent."""


class SourceEncodingError(RestrictError):
    """Raised when a source file is not plain UTF-8."""


class ObfuscationError(RestrictError):
    """Raised when the display copy does not keep the rows of the source."""
