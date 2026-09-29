from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class AttestationSettings(BaseSettings):
    """Which attestation provider to use, and its configuration.

    Separate from :class:`EnclaveSettings` because the attestation HTTP server
    needs only these, and requiring ``email``/``data_owners`` there would mean
    the endpoint silently degraded whenever they were absent. ``EnclaveSettings``
    inherits it, so both paths read the same environment variables.
    """

    model_config = SettingsConfigDict(
        env_prefix="SYFT_ENCLAVE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    attestation_provider: Literal["auto", "confidential_space", "tinfoil", "none"] = (
        Field(
            default="auto",
            description=(
                "Which deployment target to collect attestation evidence from. "
                "'auto' probes each provider's marker path; 'none' disables "
                "attestation entirely (local development)."
            ),
        )
    )
    tinfoil_repo: Optional[str] = Field(
        default=None,
        description=(
            "Tinfoil config repo ('owner/name') whose signed release published "
            "this enclave's expected measurement. Recorded in the published "
            "evidence so a verifier can warn on a mismatch; verifiers must "
            "still supply their own, since this value is enclave-controlled."
        ),
    )
    tinfoil_release_tag: Optional[str] = Field(
        default=None,
        description=(
            "Tinfoil config release tag this enclave was deployed from, e.g. "
            "'v0.1.3'. Recorded in the published evidence, as above."
        ),
    )
    tinfoil_host: Optional[str] = Field(
        default=None,
        description=(
            "Public hostname peers can reach this enclave on, e.g. "
            "'syft-enclave.openmined.containers.tinfoil.dev'. Published in the "
            "evidence so a peer knows where to fetch the attestation over a "
            "pinned connection. Untrusted: a wrong host either fails the pin "
            "or is the right enclave."
        ),
    )


class EnclaveSettings(AttestationSettings):
    """Runtime configuration for ``python -m syft_enclaves``.

    Every field maps to an environment variable with a ``SYFT_ENCLAVE_``
    prefix (e.g. ``email`` is read from ``SYFT_ENCLAVE_EMAIL``). Values may
    also be placed in a ``.env`` file in the working directory.

    Example ``.env`` for local development::

        SYFT_ENCLAVE_EMAIL=enclave@openmined.org
        SYFT_ENCLAVE_DATA_OWNERS=do1@openmined.org,do2@openmined.org
        SYFT_ENCLAVE_TOKEN_PATH=/secrets/gdrive_token.json   # optional
        SYFT_ENCLAVE_REQUIRE_TEE=false
    """

    model_config = SettingsConfigDict(
        env_prefix="SYFT_ENCLAVE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    email: str = Field(
        description="Email address of the enclave datasite. Required.",
    )
    # NoDecode: keep pydantic-settings from JSON-parsing the env value so the
    # validator below can split the comma-separated string itself.
    data_owners: Annotated[list[str], NoDecode] = Field(
        description=(
            "Emails of the data owners whose approval gates every job on this "
            "enclave. A job runs only after ALL of them approve. Accepts a "
            "comma-separated string (e.g. 'do1@x.com,do2@y.com'). Required."
        ),
    )

    @field_validator("data_owners", mode="before")
    @classmethod
    def _split_data_owners(cls, v: object) -> object:
        """Allow a comma-separated string (as passed via VM metadata)."""
        if isinstance(v, str):
            return [email.strip() for email in v.split(",") if email.strip()]
        return v

    # Default coupled with docker/entrypoint.sh, which writes the
    # operator-supplied token to this exact path before the runner starts.
    token_path: Path = Field(
        default=Path("/run/syft-enclave/token.json"),
        description=(
            "Filesystem path to a pre-authorized Google Drive OAuth token. "
            "In production, docker/entrypoint.sh writes the token content "
            "shipped via SYFT_ENCLAVE_TOKEN_CONTENT to this path before the "
            "runner starts."
        ),
    )
    poll_interval: int = Field(
        default=1,
        ge=1,
        description="Seconds to wait between poll-loop cycles.",
    )
    require_tee: bool = Field(
        default=False,
        description=(
            "Refuse to start unless an attestation provider detects its TEE. "
            "Set true in production, false for local testing."
        ),
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Root logging level.",
    )
    fresh_state: bool = Field(
        default=True,
        description=(
            "If true (default), wipe all enclave state (local SyftBox folder + "
            "Google Drive files) before init. Ensures a clean slate every boot — "
            "peers must re-add the enclave. Set false to preserve state across "
            "restarts (e.g., for stateful job continuation)."
        ),
    )
    receipts: bool = Field(
        default=False,
        description=(
            "Write a receipt signed with the enclave's identity key into each "
            "job's outputs. Pinned in the measured Tinfoil config, so a data "
            "owner can see from the release whether receipts are on."
        ),
    )
    use_encryption: bool = Field(
        default=True,
        description=(
            "End-to-end drive encryption for all enclave peer communication. "
            "Enabled by default; set false to disable."
        ),
    )
