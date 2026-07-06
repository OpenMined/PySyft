"""Registry-level schema computation and historic registration/lookup."""

import pytest

from syft_migration import (
    MigrationError,
    PackageInfo,
    ProtocolSchema,
    ReleaseArtifact,
)

from mocks import JobV1


def test_current_protocol_schema_is_computed_and_history_stored(registry):
    # Current schema is computed from the registered objects.
    current = registry.compute_protocol_schema()
    assert current.version == "2"
    assert current.supported_versions == {"job": ["1", "2", "3"]}
    assert current.current_schema(canonical_name="job") == "3"
    # History is keyed by the protocol version the past release spoke.
    historic_info = registry.package_version_history["1"]
    assert historic_info.version == "1.0.0"
    historic = registry.schema_for_protocol_version(protocol_version="1")
    assert historic.supported_versions == {"job": ["1"]}
    assert registry.latest_version(canonical_name="job") == "3"


def test_register_historic_release_artifact(registry):
    artifact = ReleaseArtifact(
        package_info=PackageInfo(
            package_name="syft-mock", version="1.0.5", protocol_version="1"
        ),
        protocol_schema=ProtocolSchema.from_objects(
            protocol_name="mock-proto",
            version="1",
            classes=[JobV1],
        ),
    )
    registry.register_historic_release_artifact(artifact=artifact)
    # Both histories are keyed by the protocol version the release spoke.
    assert registry.package_version_history["1"] is artifact.package_info
    assert registry.protocol_version_history["1"] is artifact.protocol_schema


def test_schema_for_protocol_version(registry):
    # The current protocol version resolves to the computed schema.
    current = registry.schema_for_protocol_version(protocol_version="2")
    assert current.supported_versions == {"job": ["1", "2", "3"]}
    with pytest.raises(MigrationError):
        registry.schema_for_protocol_version(protocol_version="99")
