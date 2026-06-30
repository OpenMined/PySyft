import pytest

from syft_migration import (
    MigratableObject,
    MigrationError,
    MigrationRegistry,
    PackageProtocolSchema,
)

from mocks import JobV1, JobV2, JobV3


def test_subclass_auto_registers(registry):
    assert registry.get_class("job", "1") is JobV1
    assert registry.get_class("job", "2") is JobV2
    assert set(registry.versions("job")) == {"1", "2", "3"}


def test_protocol_schema_stored_as_current_and_history(registry):
    assert registry.current_protocol_schema.package_version == "1.1.0"
    assert registry.current_protocol_schema.objects == {"job": "2"}
    assert registry.history_protocol_schemas["1.0.0"].objects == {"job": "1"}
    assert registry.latest_version("job") == "2"


def test_register_schema_adds_objects_and_syncs():
    reg = MigrationRegistry()

    class WidgetV1(MigratableObject, registry=reg):
        canonical_name: str = "widget"
        version: str = "1"

    # Object known, but not yet part of any schema (the "rare case").
    assert reg.get_class("widget", "1") is WidgetV1
    assert reg.current_protocol_schema is None

    schema = PackageProtocolSchema.from_objects("p", "pkg", "1.0.0", [WidgetV1])
    reg.register_protocol_schema(schema)
    assert reg.current_protocol_schema is schema


def test_register_schema_raises_on_conflicting_object():
    reg = MigrationRegistry()

    class AlphaV1(MigratableObject, registry=reg):
        canonical_name: str = "thing"
        version: str = "1"

    class BetaV1(MigratableObject, registry=reg.__class__()):
        # Different class with the same identity, registered into a throwaway registry.
        canonical_name: str = "thing"
        version: str = "1"

    schema = PackageProtocolSchema.from_objects("p", "pkg", "1.0.0", [BetaV1])
    with pytest.raises(MigrationError):
        reg.register_protocol_schema(schema)


def test_register_object_idempotent_for_same_class(registry):
    registry.register_object(JobV1)  # already registered; must not raise
    assert registry.get_class("job", "1") is JobV1


def test_migrate_upgrades_single_step(service):
    result = service.migrate(JobV1(name="x"), "2")
    assert isinstance(result, JobV2)
    assert result.name == "x" and result.owner == "unknown"


def test_migrate_chains_multiple_steps(service):
    result = service.migrate(JobV1(name="x"), "3")
    assert isinstance(result, JobV3)
    assert result.version == "3"


def test_migrate_downgrades(service):
    result = service.migrate(JobV2(name="x", owner="bob"), "1")
    assert isinstance(result, JobV1)
    assert result.name == "x"


def test_migrate_same_version_is_noop(service):
    job = JobV2(name="x")
    assert service.migrate(job, "2") is job


def test_migrate_missing_path_raises(service):
    with pytest.raises(MigrationError):
        service.migrate(JobV3(name="x"), "1")


def test_downgrade_for_package_version(service):
    result = service.downgrade_for_package_version(
        JobV2(name="x", owner="bob"), "1.0.0"
    )
    assert isinstance(result, JobV1)
    assert result.name == "x"


def test_load_deserializes_into_historical_class(service):
    obj = service.load({"canonical_name": "job", "version": "1", "name": "x"})
    assert isinstance(obj, JobV1)
    migrated = service.migrate(obj, service.registry.latest_version("job"))
    assert isinstance(migrated, JobV2)


def test_load_unknown_object_raises(service):
    with pytest.raises(MigrationError):
        service.load({"canonical_name": "job", "version": "99"})


def test_schema_save_load_roundtrip(tmp_path):
    schema = PackageProtocolSchema.from_objects(
        "mock-proto", "syft-mock", "1.1.0", [JobV2]
    )
    path = tmp_path / "protocol.json"
    schema.save(path)
    loaded = PackageProtocolSchema.load(path)
    assert loaded.protocol_name == "mock-proto"
    assert loaded.package_name == "syft-mock"
    assert loaded.package_version == "1.1.0"
    assert loaded.objects == {"job": "2"}
    assert loaded.object_classes() == []  # class refs are not serialized


def test_schema_rejects_two_versions_of_same_object():
    with pytest.raises(MigrationError):
        PackageProtocolSchema.from_objects("p", "pkg", "1.0.0", [JobV1, JobV2])
