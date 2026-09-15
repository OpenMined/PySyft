"""Appraising an enclave's attestation evidence.

The client side of the seam. One verifier per deployment target, both reporting
through the same checklist:

- :mod:`~syft_enclaves.attestation.confidential_space` — a Google-signed JWT.
- :mod:`~syft_enclaves.attestation.tinfoil` — a SEV-SNP/TDX report, appraised
  against the measurement a config repo published, over a connection pinned to
  the TLS key that report commits to.

:mod:`~syft_enclaves.attestation.envelope` is the wire format both targets
share; :mod:`~syft_enclaves.attestation.dispatch` routes evidence to the
verifier for its kind. The producing side lives in
:mod:`syft_enclaves.evidence`.

Re-exported here so ``from syft_enclaves.attestation import ...`` keeps working
for callers that predate this package.
"""

from syft_enclaves.attestation.confidential_space import (
    ATTESTATION_AUDIENCE,
    CONFIDENTIAL_COMPUTING_CERTS_URL,
    JWT_EXPIRY_GRACE_SECONDS,
    AppraisalPolicy,
    verify_attestation_token,
)
from syft_enclaves.attestation.dispatch import policy_for, verify_evidence
from syft_enclaves.attestation.envelope import (
    AttestationEvidence,
    AttestationKind,
    confidential_space_evidence,
    tinfoil_evidence,
)
from syft_enclaves.attestation.result import (
    AttestationError,
    AttestationResult,
    CheckResult,
)
from syft_enclaves.attestation.tinfoil import (
    TinfoilAppraisalPolicy,
    verify_tinfoil_evidence,
)

__all__ = [
    "ATTESTATION_AUDIENCE",
    "CONFIDENTIAL_COMPUTING_CERTS_URL",
    "JWT_EXPIRY_GRACE_SECONDS",
    "AppraisalPolicy",
    "AttestationError",
    "AttestationEvidence",
    "AttestationKind",
    "AttestationResult",
    "CheckResult",
    "TinfoilAppraisalPolicy",
    "confidential_space_evidence",
    "policy_for",
    "tinfoil_evidence",
    "verify_attestation_token",
    "verify_evidence",
    "verify_tinfoil_evidence",
]
