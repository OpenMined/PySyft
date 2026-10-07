"""Provider-agnostic attestation evidence carried in ``SYFT_version.json``.

``syft`` knows nothing about attestation: it carries an opaque ``extra`` bag on
``VersionInfo``, and this package owns the ``"attestation"`` key inside it
along with everything under it. Both TEE providers use the same envelope:

- ``format`` names the evidence type, as a predicate URI.
- ``body`` is the opaque evidence: a Google-signed JWT for Confidential Space, a
  base64 gzip hardware report for Tinfoil.
- ``metadata`` carries whatever the verifier of that kind needs to locate its
  reference values (for Tinfoil, which config repo and release published the
  expected measurement).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class AttestationKind(str, Enum):
    """Which TEE produced the evidence, and therefore which verifier reads it."""

    CONFIDENTIAL_SPACE = "confidential_space"
    TINFOIL = "tinfoil"


# Confidential Space has no predicate URI of its own — the evidence is an OIDC
# token — so syft names the format it publishes.
CONFIDENTIAL_SPACE_FORMAT = (
    "https://syft.openmined.org/predicate/gcp-confidential-space/v1"
)
# Tinfoil's formats come from the enclave's own attestation document; see
# https://docs.tinfoil.sh/verification/predicate.
TINFOIL_FORMAT_PREFIX = "https://tinfoil.sh/predicate/"

#: The key this package owns in ``VersionInfo.extra``. Nothing in ``syft``
#: refers to it; it is defined and read here only.
EXTRA_KEY = "attestation"

_FORMATS: dict[AttestationKind, tuple[str, ...]] = {
    AttestationKind.CONFIDENTIAL_SPACE: (CONFIDENTIAL_SPACE_FORMAT,),
    AttestationKind.TINFOIL: (TINFOIL_FORMAT_PREFIX,),
}


class AttestationEvidence(BaseModel):
    """What an enclave publishes about itself, before anyone has verified it.

    Untrusted until appraised: every field here is written by the enclave (or by
    whoever controls its transport), so a verifier must treat ``metadata`` as a
    hint and never as a source of trust. In particular the Tinfoil config repo
    must come from the verifier's own policy, since it decides *who was allowed
    to produce the expected measurement*.
    """

    model_config = {"frozen": True}

    schema_version: int = 1
    kind: AttestationKind
    format: str
    body: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("body")
    @classmethod
    def _body_is_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("attestation evidence has an empty body")
        return v

    @model_validator(mode="after")
    def _format_matches_kind(self) -> "AttestationEvidence":
        """Reject a ``kind`` that disagrees with ``format``.

        Without this, a peer could label a Confidential Space token as Tinfoil
        evidence (or vice versa) and pick which verifier appraises it.
        """
        allowed = _FORMATS[self.kind]
        if not any(self.format.startswith(prefix) for prefix in allowed):
            raise ValueError(
                f"format {self.format!r} is not valid for kind {self.kind.value!r} "
                f"(expected one of {allowed})"
            )
        return self

    def to_version_field(self) -> dict[str, Any]:
        """The JSON object stored under ``VersionInfo.extra["attestation"]``."""
        return self.model_dump(mode="json")

    @classmethod
    def from_version_field(
        cls, value: Optional[dict[str, Any]]
    ) -> Optional["AttestationEvidence"]:
        """Parse one stored envelope; ``None`` when the peer published none.

        Raises ``ValueError`` on a malformed envelope rather than returning
        ``None``: a peer that published *something* unparseable must not be
        treated the same as a peer that published nothing, which callers skip.
        """
        if value is None:
            return None
        return cls.model_validate(value)

    def publish_to(self, version_info: Any) -> None:
        """Store this evidence in a ``VersionInfo``'s extra bag."""
        version_info.extra[EXTRA_KEY] = self.to_version_field()

    @classmethod
    def read_from(cls, version_info: Any) -> Optional["AttestationEvidence"]:
        """The evidence a ``VersionInfo`` carries, or ``None`` if it carries none."""
        return cls.from_version_field((version_info.extra or {}).get(EXTRA_KEY))


def confidential_space_evidence(
    token: str,
    audience: str,
    claims: Optional[dict[str, Any]] = None,
) -> AttestationEvidence:
    """Wrap a Confidential Space attestation JWT.

    ``claims`` are the runtime facts the token's nonce commits to — the
    enclave's email, its data owners and its key bundle. Carried in the clear
    and untrusted: the digest inside the signed token is what makes them true.
    See ``attestation.claims``.
    """
    metadata: dict[str, Any] = {"audience": audience}
    if claims is not None:
        metadata["claims"] = claims
    return AttestationEvidence(
        kind=AttestationKind.CONFIDENTIAL_SPACE,
        format=CONFIDENTIAL_SPACE_FORMAT,
        body=token,
        metadata=metadata,
    )


def tinfoil_evidence(
    document: dict[str, Any],
    repo: Optional[str] = None,
    release_tag: Optional[str] = None,
    host: Optional[str] = None,
) -> AttestationEvidence:
    """Wrap a Tinfoil ``{format, body}`` attestation document.

    ``repo``/``release_tag`` are recorded so a verifier can warn when its own
    policy points somewhere else; they never widen trust. ``host`` tells a peer
    where to fetch the report over a pinned connection — also untrusted, since
    a wrong host either fails the pin or is the right enclave.
    """
    try:
        format_, body = document["format"], document["body"]
    except (KeyError, TypeError) as e:
        raise ValueError(
            "tinfoil attestation document must have 'format' and 'body' keys"
        ) from e
    metadata = {
        key: value
        for key, value in (("repo", repo), ("release_tag", release_tag), ("host", host))
        if value
    }
    return AttestationEvidence(
        kind=AttestationKind.TINFOIL,
        format=format_,
        body=body,
        metadata=metadata,
    )
