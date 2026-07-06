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

from mocks import JobV1, JobV2, JobV3


def test_subclass_auto_registers(registry):
    assert registry.get_class(canonical_name="job", version="1") is JobV1
    assert registry.get_class(canonical_name="job", version="2") is JobV2
    assert set(registry.versions(canonical_name="job")) == {"1", "2", "3"}


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


def test_concrete_class_without_registry_raises():
    with pytest.raises(MigrationError):

        class OrphanV1(MigratableObject):
            canonical_name: str = "orphan"
            version: str = "1"


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


def test_downgrade_for_protocol_version(service):
    result = service.downgrade_for_protocol_version(
        obj=JobV2(name="x", owner="bob"), protocol_version="1"
    )
    assert isinstance(result, JobV1)
    assert result.name == "x"


def test_has_upgradeable_path_to_latest(registry):
    # job has migrations 1 -> 2 -> 3; the latest version is trivially upgradeable.
    assert registry.has_upgradeable_path_to_latest("job", "1")
    assert registry.has_upgradeable_path_to_latest("job", "2")
    assert registry.has_upgradeable_path_to_latest("job", "3")


def test_has_upgradeable_path_single_version_and_missing_migration():
    reg = MigrationRegistry(
        protocol_name="p",
        package_name="pkg",
        package_version="1.0.0",
        protocol_version="1",
    )

    class ThingV1(MigratableObject, registry=reg):
        canonical_name: str = "thing"
        version: str = "1"

    # A single registered version is trivially upgradeable to itself.
    assert reg.has_upgradeable_path_to_latest("thing", "1")

    class ThingV2(MigratableObject, registry=reg):
        canonical_name: str = "thing"
        version: str = "2"

    # Two versions without a registered migration: no path.
    assert not reg.has_upgradeable_path_to_latest("thing", "1")

    reg.register_migration(
        canonical_name="thing",
        from_version="1",
        to_version="2",
        fn=lambda obj: ThingV2(),
    )
    assert reg.has_upgradeable_path_to_latest("thing", "1")


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
    schema = ProtocolSchema.from_objects(
        protocol_name="mock-proto",
        version="2",
        classes=[JobV2],
    )
    path = tmp_path / "protocol.json"
    schema.save(path=path)
    loaded = ProtocolSchema.load(path=path)
    assert loaded.protocol_name == "mock-proto"
    assert loaded.version == "2"
    assert loaded.supported_versions == {"job": ["2"]}


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


def test_registry_computes_protocol_schema(registry):
    protocol = registry.compute_protocol_schema()
    assert protocol.protocol_name == "mock-proto"
    assert protocol.version == "2"
    assert protocol.supported_versions == {"job": ["1", "2", "3"]}


def test_schema_collects_all_versions_of_same_object():
    schema = ProtocolSchema.from_objects(
        protocol_name="p",
        version="1",
        classes=[JobV2, JobV1],
    )
    assert schema.supported_versions == {"job": ["1", "2"]}
    assert schema.current_schema(canonical_name="job") == "2"


def test_current_schema_raises_on_unknown_object():
    schema = ProtocolSchema.from_objects(
        protocol_name="p",
        version="1",
        classes=[JobV1],
    )
    with pytest.raises(MigrationError):
        schema.current_schema(canonical_name="unknown")


def test_protocol_schema_save_load_roundtrip(tmp_path):
    protocol = ProtocolSchema(
        protocol_name="mock-proto",
        version="2",
        supported_versions={"job": ["1", "2"]},
    )
    path = tmp_path / "protocol_schema.json"
    protocol.save(path=path)
    assert ProtocolSchema.load(path=path) == protocol


def test_release_artifact_save_load_roundtrip(tmp_path):
    artifact = ReleaseArtifact(
        package_info=PackageInfo(
            package_name="syft-mock", version="1.0.0", protocol_version="1"
        ),
        protocol_schema=ProtocolSchema.from_objects(
            protocol_name="mock-proto",
            version="1",
            classes=[JobV1],
        ),
    )
    path = tmp_path / "release.json"
    artifact.save(path=path)
    assert ReleaseArtifact.load(path=path) == artifact


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
