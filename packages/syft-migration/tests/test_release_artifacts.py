"""Release artifacts and the protocol-version history they register."""

import pytest

from syft_migration import (
    MigratableObject,
    MigrationError,
    MigrationRegistry,
    PackageInfo,
    ProtocolSchema,
    ReleasedPackageProtocolInfo,
)

from mocks import MyVersionedObjectV1, MyVersionedObjectV3


def _fresh_registry(protocol_version: str = "1") -> MigrationRegistry:
    return MigrationRegistry(
        protocol_name="mock-proto",
        package_name="syft-mock",
        package_version="1.1.0",
        protocol_version=protocol_version,
    )


def test_release_artifact_save_load_roundtrip(tmp_path):
    artifact = ReleasedPackageProtocolInfo(
        package_info=PackageInfo(
            package_name="syft-mock", version="1.0.0", protocol_version="1"
        ),
        protocol_schema=ProtocolSchema.from_objects(
            protocol_name="mock-proto",
            version="1",
            classes=[MyVersionedObjectV1],
        ),
    )
    path = tmp_path / "release.json"
    artifact.save(path=path)
    assert ReleasedPackageProtocolInfo.load(path=path) == artifact


def test_register_released_package_protocol_info():
    # A fresh registry, so the shared mock registry's history stays untouched.
    reg = MigrationRegistry(
        protocol_name="mock-proto",
        package_name="syft-mock",
        package_version="1.1.0",
        protocol_version="2",
    )
    artifact = ReleasedPackageProtocolInfo(
        package_info=PackageInfo(
            package_name="syft-mock", version="1.0.5", protocol_version="1"
        ),
        protocol_schema=ProtocolSchema.from_objects(
            protocol_name="mock-proto",
            version="1",
            classes=[MyVersionedObjectV1],
        ),
    )
    reg.register_released_package_protocol_info(info=artifact)
    # Both histories are keyed by the protocol version the release spoke.
    assert reg.package_version_history["1"] is artifact.package_info
    assert reg.protocol_version_history["1"] is artifact.protocol_schema


def test_current_protocol_schema_is_computed_and_old_protocols_stored(registry):
    # Current schema is computed from the registered objects.
    current = registry.compute_protocol_schema()
    assert current.version == "2"
    assert current.supported_versions == {"MyVersionedObject": ["1", "2", "3"]}
    assert current.current_schema(canonical_name="MyVersionedObject") == "3"
    # History is keyed by the protocol version the past release spoke.
    historic_info = registry.package_version_history["1"]
    assert historic_info.version == "1.0.0"
    historic = registry.schema_for_protocol_version(protocol_version="1")
    assert historic.supported_versions == {"MyVersionedObject": ["1"]}
    assert registry.latest_version(canonical_name="MyVersionedObject") == "3"


def test_schema_for_protocol_version(registry):
    # The current protocol version resolves to the computed schema.
    current = registry.schema_for_protocol_version(protocol_version="2")
    assert current.supported_versions == {"MyVersionedObject": ["1", "2", "3"]}
    with pytest.raises(MigrationError):
        registry.schema_for_protocol_version(protocol_version="99")


def test_compute_released_protocol(registry, tmp_path):
    released = registry.compute_released_protocol()

    # The protocol freezes the schema of each object's CURRENT (latest) version.
    schema = released.protocol_schema
    assert schema.current_schema("MyVersionedObject") == "3"
    assert (
        schema.current_object_schemas["MyVersionedObject"]
        == MyVersionedObjectV3.model_json_schema()
    )

    path = tmp_path / "protocol-2.json"
    released.save(path=path)
    assert type(released).load(path=path) == released

    # Registering a protocol computed from the same classes shows no drift.
    registry.register_released_protocol(released=released)
    assert registry.find_schema_drift() == []


def test_find_schema_drift_flags_tampered_class():
    release_registry = _fresh_registry()

    class ThingV1(MigratableObject, registry=release_registry):
        canonical_name: str = "thing"
        version: str = "1"
        name: str = ""

    released = release_registry.compute_released_protocol()

    # A later codebase where someone edited the released class in place.
    tampered_registry = _fresh_registry()

    class TamperedThingV1(MigratableObject, registry=tampered_registry):
        canonical_name: str = "thing"
        version: str = "1"
        name: str = ""
        sneaky_extra_field: int = 0

    tampered_registry.register_released_protocol(released=released)
    assert tampered_registry.find_schema_drift() == [("thing", "1", "1")]

    # A codebase that dropped the released class entirely also drifts.
    missing_registry = _fresh_registry()
    missing_registry.register_released_protocol(released=released)
    assert missing_registry.find_schema_drift() == [("thing", "1", "1")]


def test_protocol_changed_without_bump():
    reg = _fresh_registry()

    class GadgetV1(MigratableObject, registry=reg):
        canonical_name: str = "gadget"
        version: str = "1"

    # No released protocol for this registry's version yet: nothing to compare.
    assert not reg.protocol_changed_without_bump()

    reg.register_released_protocol(released=reg.compute_released_protocol())
    assert not reg.protocol_changed_without_bump()

    # A new object version changes the protocol; the version constant did not move.
    class GadgetV2(MigratableObject, registry=reg):
        canonical_name: str = "gadget"
        version: str = "2"

    assert reg.protocol_changed_without_bump()
