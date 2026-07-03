import pytest

from syft_migration import (
    MigratableObject,
    MigrationError,
    MigrationRegistry,
    MigrationService,
    PackageProtocolSchema,
    ProtocolSchema,
)

from mocks import JobV1, JobV2, JobV3


def test_subclass_auto_registers(registry):
    assert registry.get_class(canonical_name="job", version="1") is JobV1
    assert registry.get_class(canonical_name="job", version="2") is JobV2
    assert set(registry.versions(canonical_name="job")) == {"1", "2", "3"}


def test_current_protocol_schema_is_computed_and_history_stored(registry):
    # Current schema is computed from the registered objects: latest of each.
    assert registry.current_protocol_schema.package_version == "1.1.0"
    assert registry.current_protocol_schema.object_versions == {"job": "3"}
    assert registry.history_protocol_schemas["1.0.0"].object_versions == {"job": "1"}
    assert registry.latest_version(canonical_name="job") == "3"


def test_current_protocol_schema_tracks_newly_defined_objects():
    reg = MigrationRegistry(
        protocol_name="p", package_name="pkg", package_version="1.0.0"
    )
    assert reg.current_protocol_schema.object_versions == {}

    class WidgetV1(MigratableObject, registry=reg):
        canonical_name: str = "widget"
        version: str = "1"

    assert reg.get_class(canonical_name="widget", version="1") is WidgetV1
    assert reg.current_protocol_schema.object_versions == {"widget": "1"}


def test_register_historic_schema_raises_on_unregistered_object():
    reg = MigrationRegistry(
        protocol_name="p", package_name="pkg", package_version="1.0.0"
    )
    throwaway = MigrationRegistry(
        protocol_name="p", package_name="pkg", package_version="0.9.0"
    )

    class BetaV1(MigratableObject, registry=throwaway):
        # Registered into a throwaway registry, so ``reg`` has never seen it.
        canonical_name: str = "thing"
        version: str = "1"

    schema = PackageProtocolSchema.from_objects(
        protocol_name="p",
        package_name="pkg",
        package_version="0.9.0",
        classes=[BetaV1],
    )
    with pytest.raises(MigrationError):
        reg.register_historic_protocol_schema(schema=schema)


def test_register_object_version_idempotent_for_same_class(registry):
    registry.register_object_version(cls=JobV1)  # already registered; must not raise
    assert registry.get_class(canonical_name="job", version="1") is JobV1


def test_migrate_upgrades_single_step(service):
    result = service.migrate(obj=JobV1(name="x"), target_version="2")
    assert isinstance(result, JobV2)
    assert result.name == "x" and result.owner == "unknown"


def test_migrate_chains_multiple_steps(service):
    result = service.migrate(obj=JobV1(name="x"), target_version="3")
    assert isinstance(result, JobV3)
    assert result.version == "3"


def test_migrate_downgrades(service):
    result = service.migrate(obj=JobV2(name="x", owner="bob"), target_version="1")
    assert isinstance(result, JobV1)
    assert result.name == "x"


def test_migrate_same_version_is_noop(service):
    job = JobV2(name="x")
    assert service.migrate(obj=job, target_version="2") is job


def test_migrate_missing_path_raises(service):
    with pytest.raises(MigrationError):
        service.migrate(obj=JobV3(name="x"), target_version="1")


def test_downgrade_for_package_version(service):
    result = service.downgrade_for_package_version(
        obj=JobV2(name="x", owner="bob"), package_version="1.0.0"
    )
    assert isinstance(result, JobV1)
    assert result.name == "x"


def test_load_deserializes_into_historical_class(service):
    obj = service.load(data={"canonical_name": "job", "version": "1", "name": "x"})
    assert isinstance(obj, JobV1)
    migrated = service.migrate(
        obj=obj, target_version=service.registry.latest_version(canonical_name="job")
    )
    assert isinstance(migrated, JobV3)


def test_load_unknown_object_raises(service):
    with pytest.raises(MigrationError):
        service.load(data={"canonical_name": "job", "version": "99"})


def test_schema_save_load_roundtrip(tmp_path):
    schema = PackageProtocolSchema.from_objects(
        protocol_name="mock-proto",
        package_name="syft-mock",
        package_version="1.1.0",
        classes=[JobV2],
    )
    path = tmp_path / "protocol.json"
    schema.save(path=path)
    loaded = PackageProtocolSchema.load(path=path)
    assert loaded.protocol_name == "mock-proto"
    assert loaded.package_name == "syft-mock"
    assert loaded.package_version == "1.1.0"
    assert loaded.object_versions == {"job": "2"}


def _multi_object_registry() -> MigrationRegistry:
    """A registry with two canonical names and one historic release."""
    reg = MigrationRegistry(
        protocol_name="mock-proto",
        package_name="syft-mock",
        package_version="1.1.0",
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

    historic = PackageProtocolSchema.from_objects(
        protocol_name="mock-proto",
        package_name="syft-mock",
        package_version="1.0.0",
        classes=[DatasetV1, ModelV1],
    )
    reg.register_historic_protocol_schema(schema=historic)
    return reg


def test_service_exports_protocol_schema_with_all_supported_versions():
    service = MigrationService(registry=_multi_object_registry())
    protocol = service.export_protocol_schema()

    assert isinstance(protocol, ProtocolSchema)
    assert protocol.protocol_name == "mock-proto"
    # The version of the package release that produced the export.
    assert protocol.version == "1.1.0"
    # All registered versions per canonical name, not just the pinned ones.
    assert protocol.supported_versions == {"dataset": ["1", "2"], "model": ["1"]}


def test_registry_computes_protocol_schema(registry):
    protocol = registry.compute_protocol_schema()
    assert protocol.protocol_name == "mock-proto"
    assert protocol.version == "1.1.0"
    assert protocol.supported_versions == {"job": ["1", "2", "3"]}


def test_schema_rejects_two_versions_of_same_object():
    with pytest.raises(MigrationError):
        PackageProtocolSchema.from_objects(
            protocol_name="p",
            package_name="pkg",
            package_version="1.0.0",
            classes=[JobV1, JobV2],
        )
