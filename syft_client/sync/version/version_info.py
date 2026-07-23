"""
VersionInfo model for representing version information.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import Field
from syft_migration import MigratableObject, ProtocolSchema

from syft_client.migrations import client_registry, load_as_latest
from syft_client.version import (
    MIN_SUPPORTED_PROTOCOL_VERSION,
    MIN_SUPPORTED_SYFT_CLIENT_VERSION,
    PROTOCOL_VERSION,
    SYFT_CLIENT_VERSION,
)


class CompatibilityStatus(str, Enum):
    """Outcome of comparing two VersionInfo objects."""

    SAME = "same"
    PATCH_DIFF = "patch_diff"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


def _parse_semver(version_str: str) -> tuple[int, int, int]:
    """Parse 'X.Y.Z' into (major, minor, patch). Raise ValueError if not parseable."""
    parts = version_str.split(".")
    if len(parts) < 3:
        raise ValueError(f"Invalid semver: {version_str!r} (expected 'X.Y.Z')")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


class VersionInfoV1(MigratableObject, registry=client_registry):
    """Model representing version information for a syft client.

    Stored as SYFT_version.json in the peer-visible SyftBox folder. This file
    is the bootstrap channel for protocol negotiation (peers read it to learn
    what we speak), so its schema may only ever change additively: every
    supported client version must be able to parse every newer version file.
    """

    canonical_name: str = "VersionInfo"
    version: str = "1"

    syft_client_version: str
    min_supported_syft_client_version: str
    protocol_version: str
    min_supported_protocol_version: str
    syft_client_install_source: Optional[str] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    attestation_token: Optional[str] = None

    def compatibility_status_with(
        self, other: "VersionInfo" | None
    ) -> CompatibilityStatus:
        """Compare syft_client_version against another VersionInfo."""
        if other is None:
            return CompatibilityStatus.UNKNOWN

        if self.syft_client_version == other.syft_client_version:
            return CompatibilityStatus.SAME

        local = _parse_semver(self.syft_client_version)
        peer = _parse_semver(other.syft_client_version)

        if local[0] == peer[0] and local[1] == peer[1]:
            return CompatibilityStatus.PATCH_DIFF

        return CompatibilityStatus.INCOMPATIBLE

    def is_compatible_with(
        self,
        other: "VersionInfo" | None,
        compatible_if_unknown: bool = False,
    ) -> bool:
        """True if SAME or PATCH_DIFF (patch differences are non-blocking).

        If `other` is None, returns `compatible_if_unknown`.
        """
        status = self.compatibility_status_with(other)
        if status == CompatibilityStatus.UNKNOWN:
            return compatible_if_unknown
        return status in (CompatibilityStatus.SAME, CompatibilityStatus.PATCH_DIFF)

    def get_incompatibility_reason(self, other: "VersionInfo") -> Optional[str]:
        """Reason string when minor/major mismatch; None for SAME or PATCH_DIFF."""
        status = self.compatibility_status_with(other)
        if status in (CompatibilityStatus.SAME, CompatibilityStatus.PATCH_DIFF):
            return None
        return (
            f"Client version mismatch (minor or major): "
            f"local={self.syft_client_version}, peer={other.syft_client_version}"
        )

    def get_patch_warning_text(self, other: "VersionInfo") -> Optional[str]:
        """Warning string when only patch versions differ."""
        status = self.compatibility_status_with(other)
        if status != CompatibilityStatus.PATCH_DIFF:
            return None
        return (
            f"Client version differs by patch only: "
            f"local={self.syft_client_version}, peer={other.syft_client_version} "
            f"— proceeding"
        )

    @classmethod
    def current(cls) -> "VersionInfo":
        """Create VersionInfo with current version constants."""
        install_source: Optional[str] = None
        try:
            from syft_job.install_source import get_syft_client_install_source

            install_source = get_syft_client_install_source()
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.debug(f"Could not detect syft-client install source: {e}")

        return cls(
            syft_client_version=SYFT_CLIENT_VERSION,
            min_supported_syft_client_version=MIN_SUPPORTED_SYFT_CLIENT_VERSION,
            protocol_version=PROTOCOL_VERSION,
            min_supported_protocol_version=MIN_SUPPORTED_PROTOCOL_VERSION,
            syft_client_install_source=install_source,
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "VersionInfo":
        """Deserialize from JSON string, upgraded to the latest version.

        Files written by protocol-0 clients (<= 0.1.117) predate the identity
        fields; they are all version 1.
        """
        return load_as_latest(json.loads(json_str), "VersionInfo")


def _slim_schema_of(registry) -> ProtocolSchema:
    """The registry's protocol schema without the embedded object JSON schemas.

    Negotiation needs only ``version`` and ``supported_versions``; skipping
    ``current_object_schemas`` keeps the published version file small (the
    full frozen schemas live in the release artifacts, not on the wire) and
    avoids computing every object's JSON schema just to discard it.
    """
    return ProtocolSchema(
        protocol_name=registry.protocol_name,
        version=registry.protocol_version,
        supported_versions={
            canonical_name: sorted(versions)
            for canonical_name, versions in registry.objects.items()
        },
    )


def _gather_protocol_schemas() -> dict[str, ProtocolSchema]:
    """Slim protocol schemas of every syft package present in this install.

    Keyed by protocol name. syft-job/syft-dataset are optional dependencies;
    a missing or broken package simply means its schema is not advertised and
    peers treat this client as an unknown speaker of that protocol (same
    failure-tolerant pattern as the install-source detection in ``current``).
    """
    logger = logging.getLogger(__name__)
    schemas = {client_registry.protocol_name: _slim_schema_of(client_registry)}
    try:
        from syft_job.migrations import job_registry

        schemas[job_registry.protocol_name] = _slim_schema_of(job_registry)
    except Exception as e:
        logger.debug(f"Not advertising a syft-job protocol schema: {e}")
    try:
        from syft_datasets.migrations.registry import dataset_registry

        schemas[dataset_registry.protocol_name] = _slim_schema_of(dataset_registry)
    except Exception as e:
        logger.debug(f"Not advertising a syft-dataset protocol schema: {e}")
    return schemas


class VersionInfoV2(VersionInfoV1):
    """V2 adds the protocol schemas this client speaks (client, job, dataset).

    Purely additive over V1 (see the bootstrap-channel rule in the V1
    docstring): protocol-0/1 readers ignore the extra key.
    """

    version: str = "2"

    # protocol name -> slim ProtocolSchema (no embedded object JSON schemas).
    protocol_schemas: dict[str, ProtocolSchema] = Field(default_factory=dict)

    @classmethod
    def current(cls) -> "VersionInfo":
        info = super().current()
        info.protocol_schemas = _gather_protocol_schemas()
        return info


@client_registry.migration("VersionInfo", "1", "2")
def _version_info_v1_to_v2(obj: VersionInfoV1) -> VersionInfoV2:
    # A v1 file says nothing about package protocols: empty schemas, meaning
    # "unknown speaker" to consumers.
    return VersionInfoV2.model_validate(
        obj.model_dump(exclude={"canonical_name", "version"})
    )


@client_registry.migration("VersionInfo", "2", "1")
def _version_info_v2_to_v1(obj: VersionInfoV2) -> VersionInfoV1:
    return VersionInfoV1.model_validate(
        obj.model_dump(exclude={"canonical_name", "version", "protocol_schemas"})
    )


# Current-version alias: callers always work with the latest VersionInfo.
VersionInfo = VersionInfoV2
