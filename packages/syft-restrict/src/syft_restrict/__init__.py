"""syft-restrict — verify + obfuscate JAX/Flax inference code.

`run` is the entry point: it statically proves the private model-definition lines only do trusted math
(no data theft), then obfuscates them so the model architecture stays secret. See the README for the
pitch, docs/verify.md for how verification works, and docs/blacklist.md for what is rejected.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .errors import PolicyViolation, RestrictError
from .obfuscator import obfuscate
from .policy import Policy
from .runner import RunResult, run
from .verifier import VerifyResult, Violation, verify

__all__ = [
    "run",
    "verify",
    "obfuscate",
    "Policy",
    "RunResult",
    "VerifyResult",
    "Violation",
    "PolicyViolation",
    "RestrictError",
    "__version__",
]
