"""Where the enclave's attestation evidence comes from.

One provider per deployment target. A provider knows how to tell whether it is
running on its own kind of TEE (``detect``), how to obtain the evidence
(``collect``), and how to summarise it for the operator-facing HTTP endpoint
(``describe``). Nothing here verifies anything — an enclave never appraises its
own evidence; that is the verifier's job (``attestation.dispatch``).

Adding a provider is: write a class with ``kind``/``probe_path``/``detect``/
``from_settings``/``collect``/``describe``, and add it to ``PROVIDERS``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

from syft_enclaves.attestation.envelope import AttestationEvidence, AttestationKind
from syft_enclaves.evidence.confidential_space import ConfidentialSpaceProvider
from syft_enclaves.evidence.tinfoil import TinfoilProvider

logger = logging.getLogger(__name__)

AUTO = "auto"
NONE = "none"


@runtime_checkable
class EvidenceProvider(Protocol):
    """How the enclave obtains evidence about itself on one deployment target."""

    kind: AttestationKind
    #: The file or socket whose presence means "we are on this TEE".
    probe_path: Path

    @classmethod
    def detect(cls) -> bool:
        """Whether this provider's TEE is the one we are running on."""

    @classmethod
    def from_settings(cls, settings: Any) -> "EvidenceProvider":
        """Build a provider from ``EnclaveSettings``.

        Each provider reads only the settings it needs, so a new target adds
        its configuration without the generic seam knowing about it.
        """

    def collect(self, caller_nonce: Optional[str] = None) -> AttestationEvidence:
        """Obtain fresh evidence, ready to publish to peers."""

    def describe(self, evidence: AttestationEvidence) -> dict[str, Any]:
        """Human-readable summary of *evidence*, for the /attestation endpoint.

        Display only, and explicitly unverified — a relying party appraises the
        evidence itself, never this summary.
        """


PROVIDERS: dict[str, type[EvidenceProvider]] = {
    AttestationKind.CONFIDENTIAL_SPACE.value: ConfidentialSpaceProvider,
    AttestationKind.TINFOIL.value: TinfoilProvider,
}


def select_provider(
    name: str = AUTO, settings: Any = None
) -> Optional[EvidenceProvider]:
    """Resolve *name* to a configured provider, or ``None`` outside a TEE.

    ``"auto"`` probes each provider in turn; ``"none"`` disables attestation
    (local development); any other value selects that provider explicitly and
    still requires it to detect, so a misconfigured deployment publishes
    nothing loudly rather than quietly.
    """
    if name == NONE:
        return None
    provider_cls = _detect_provider_class() if name == AUTO else _named(name)
    if provider_cls is None:
        return None
    if name != AUTO and not provider_cls.detect():
        logger.warning(
            "Attestation provider %r was requested but %s is not present",
            name,
            provider_cls.probe_path,
        )
        return None
    return provider_cls.from_settings(settings)


def _named(name: str) -> type[EvidenceProvider]:
    try:
        return PROVIDERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown attestation provider {name!r}. "
            f"Expected one of: {AUTO}, {NONE}, {', '.join(sorted(PROVIDERS))}."
        ) from None


def _detect_provider_class() -> Optional[type[EvidenceProvider]]:
    for name, provider_cls in PROVIDERS.items():
        if provider_cls.detect():
            logger.info("Detected attestation provider: %s", name)
            return provider_cls
    return None


def probed_locations() -> str:
    """The paths ``auto`` detection looks at, for error messages."""
    return ", ".join(f"{name} ({cls.probe_path})" for name, cls in PROVIDERS.items())
