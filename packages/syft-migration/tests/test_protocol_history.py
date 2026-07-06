"""Registry-level schema computation and historic registration/lookup."""

import pytest

from syft_migration import (
    MigratableObject,
    MigrationError,
    MigrationRegistry,
    MigrationService,
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


def test_current_protocol_schema_tracks_newly_defined_objects():
    reg = MigrationRegistry(
        protocol_name="p",
        package_name="pkg",
        package_version="1.0.0",
        protocol_version="1",
    )
    assert reg.compute_protocol_schema().supported_versions == {}

    class WidgetV1(MigratableObject, registry=reg):
        canonical_name: str = "widget"
        version: str = "1"

    assert reg.get_class(canonical_name="widget", version="1") is WidgetV1
    assert reg.compute_protocol_schema().supported_versions == {"widget": ["1"]}


def test_register_historic_schema_raises_on_unregistered_object():
    reg = MigrationRegistry(
        protocol_name="p",
        package_name="pkg",
        package_version="1.0.0",
        protocol_version="1",
    )
    throwaway = MigrationRegistry(
        protocol_name="p",
        package_name="pkg",
        package_version="0.9.0",
        protocol_version="0",
    )

    class BetaV1(MigratableObject, registry=throwaway):
        # Registered into a throwaway registry, so ``reg`` has never seen it.
        canonical_name: str = "thing"
        version: str = "1"

    schema = ProtocolSchema.from_objects(
        protocol_name="p",
        version="0",
        classes=[BetaV1],
    )
    with pytest.raises(MigrationError):
        reg.register_historic_protocol_schema(
            schema=schema, raise_for_unknown_objects=True
        )
    # Without the check the schema registers as-is.
    reg.register_historic_protocol_schema(schema=schema)
    assert reg.protocol_version_history["0"] is schema


def test_registry_computes_protocol_schema(registry):
    protocol = registry.compute_protocol_schema()
    assert protocol.protocol_name == "mock-proto"
    assert protocol.version == "2"
    assert protocol.supported_versions == {"job": ["1", "2", "3"]}


def _multi_object_registry() -> MigrationRegistry:
    """A registry with two canonical names and one historic release."""
    reg = MigrationRegistry(
        protocol_name="mock-proto",
        package_name="syft-mock",
        package_version="1.1.0",
        protocol_version="2",
    )

    class DatasetV1(MigratableObject, registry=reg):
        canonical_name: str = "dataset"
        version: str = "1"

    class DatasetV2(MigratableObject, registry=reg):
        canonical_name: str = "dataset"
        version: str = "2"

    class ModelV1(MigratableObject, registry=reg):
        canonical_name: str = "model"
        version: str = "1"

    historic = ProtocolSchema.from_objects(
        protocol_name="mock-proto",
        version="1",
        classes=[DatasetV1, ModelV1],
    )
    reg.register_historic_protocol_schema(schema=historic)
    return reg


def test_service_exports_protocol_schema_with_all_supported_versions():
    service = MigrationService(registry=_multi_object_registry())
    protocol = service.export_protocol_schema()

    assert isinstance(protocol, ProtocolSchema)
    assert protocol.protocol_name == "mock-proto"
    # The protocol version of the release that produced the export.
    assert protocol.version == "2"
    # All registered versions per canonical name, not just the pinned ones.
    assert protocol.supported_versions == {"dataset": ["1", "2"], "model": ["1"]}


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
