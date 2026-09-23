"""Routing attestation evidence to the verifier for its deployment target.

The evidence itself says which kind it is, and the envelope refuses a ``kind``
that disagrees with its ``format`` (see ``attestation.envelope``), so a peer
cannot pick a weaker verifier for its own evidence.
"""

from __future__ import annotations

from typing import Optional, Union

from syft_enclaves.attestation.confidential_space import (
    AppraisalPolicy,
    verify_attestation_token,
)
from syft_enclaves.attestation.result import AttestationResult
from syft_enclaves.attestation.envelope import AttestationEvidence, AttestationKind

Policy = Union[AppraisalPolicy, "object"]

#: Which policy class each kind expects. Kept here rather than imported from
#: the tinfoil module so this module never pulls in the optional SDK.
_POLICY_CLASS_NAMES = {
    AttestationKind.CONFIDENTIAL_SPACE: "AppraisalPolicy",
    AttestationKind.TINFOIL: "TinfoilAppraisalPolicy",
}


def verify_evidence(
    evidence: AttestationEvidence,
    policy: Optional[Policy] = None,
    verbose: bool = True,
) -> AttestationResult:
    """Appraise *evidence* with the verifier for its kind."""
    if evidence.kind is AttestationKind.CONFIDENTIAL_SPACE:
        _require_policy_type(evidence.kind, policy, AppraisalPolicy)
        return verify_attestation_token(
            evidence.body,
            policy=policy,
            verbose=verbose,
            published_claims=evidence.metadata.get("claims"),
        )

    if evidence.kind is AttestationKind.TINFOIL:
        # Imported here so installs without the tinfoil extra can still verify
        # Confidential Space evidence.
        from syft_enclaves.attestation.tinfoil import (
            TinfoilAppraisalPolicy,
            verify_tinfoil_evidence,
        )

        _require_policy_type(evidence.kind, policy, TinfoilAppraisalPolicy)
        return verify_tinfoil_evidence(evidence, policy=policy, verbose=verbose)

    raise ValueError(f"No verifier for attestation kind {evidence.kind!r}")


def policy_for(kind: AttestationKind, **kwargs) -> Policy:
    """Build the appraisal policy class that *kind*'s verifier expects."""
    if kind is AttestationKind.CONFIDENTIAL_SPACE:
        return AppraisalPolicy(**kwargs)
    if kind is AttestationKind.TINFOIL:
        from syft_enclaves.attestation.tinfoil import TinfoilAppraisalPolicy

        return TinfoilAppraisalPolicy(**kwargs)
    raise ValueError(f"No appraisal policy for attestation kind {kind!r}")


def _require_policy_type(
    kind: AttestationKind, policy: Optional[Policy], expected: type
) -> None:
    """Refuse a policy meant for a different target.

    Silently ignoring its fields would drop the caller's pinned image digest
    and quietly weaken the appraisal.
    """
    if policy is None or isinstance(policy, expected):
        return
    raise ValueError(
        f"{type(policy).__name__} cannot appraise {kind.value} evidence — "
        f"pass a {_POLICY_CLASS_NAMES[kind]} instead."
    )
