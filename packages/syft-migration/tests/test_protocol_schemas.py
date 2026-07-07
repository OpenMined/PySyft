"""The schema model classes themselves (no registry state)."""

import pytest

from syft_migration import (
    MigratableObject,
    MigrationError,
    MigrationRegistry,
    MigrationService,
    ProtocolSchema,
)

from mocks import MyVersionedObjectV1, MyVersionedObjectV2


def test_schema_save_load_roundtrip(tmp_path):
    schema = ProtocolSchema.from_objects(
        protocol_name="mock-proto",
        version="2",
        classes=[MyVersionedObjectV2],
    )
    path = tmp_path / "protocol.json"
    schema.save(path=path)
    loaded = ProtocolSchema.load(path=path)
    assert loaded.protocol_name == "mock-proto"
    assert loaded.version == "2"
    assert loaded.supported_versions == {"MyVersionedObject": ["2"]}


def test_schema_collects_all_versions_of_same_object():
    schema = ProtocolSchema.from_objects(
        protocol_name="p",
        version="1",
        classes=[MyVersionedObjectV2, MyVersionedObjectV1],
    )
    assert schema.supported_versions == {"MyVersionedObject": ["1", "2"]}
    assert schema.current_schema(canonical_name="MyVersionedObject") == "2"


def test_current_schema_raises_on_unknown_object():
    schema = ProtocolSchema.from_objects(
        protocol_name="p",
        version="1",
        classes=[MyVersionedObjectV1],
    )
    with pytest.raises(MigrationError):
        schema.current_schema(canonical_name="unknown")


def test_protocol_schema_save_load_roundtrip(tmp_path):
    protocol = ProtocolSchema(
        protocol_name="mock-proto",
        version="2",
        supported_versions={"MyVersionedObject": ["1", "2"]},
    )
    path = tmp_path / "protocol_schema.json"
    protocol.save(path=path)
    assert ProtocolSchema.load(path=path) == protocol


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
    assert protocol.supported_versions == {"MyVersionedObject": ["1", "2", "3"]}


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
