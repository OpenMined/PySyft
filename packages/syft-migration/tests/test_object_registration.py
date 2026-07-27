"""Registering versioned objects into a registry."""

import pytest

from syft_migration import MigratableObject, MigrationError, MigrationRegistry

from mocks import MyVersionedObjectV1, MyVersionedObjectV2


def test_subclass_auto_registers(registry):
    assert (
        registry.get_class(canonical_name="MyVersionedObject", version="1")
        is MyVersionedObjectV1
    )
    assert (
        registry.get_class(canonical_name="MyVersionedObject", version="2")
        is MyVersionedObjectV2
    )
    assert set(registry.versions(canonical_name="MyVersionedObject")) == {"1", "2", "3"}


def test_concrete_class_without_registry_raises():
    with pytest.raises(MigrationError):

        class OrphanV1(MigratableObject):
            canonical_name: str = "orphan"
            version: str = "1"


def test_register_object_version_idempotent_for_same_class(registry):
    registry.register_object_version(
        cls=MyVersionedObjectV1
    )  # already registered; must not raise
    assert (
        registry.get_class(canonical_name="MyVersionedObject", version="1")
        is MyVersionedObjectV1
    )


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


def test_has_upgradeable_path_to_latest(registry):
    # MyVersionedObject has migrations 1 -> 2 -> 3; latest is trivially upgradeable.
    assert registry.has_upgradeable_path_to_latest("MyVersionedObject", "1")
    assert registry.has_upgradeable_path_to_latest("MyVersionedObject", "2")
    assert registry.has_upgradeable_path_to_latest("MyVersionedObject", "3")


def test_has_migration_path(registry):
    # MyVersionedObject has migrations 1 -> 2 -> 3 and 2 -> 1, but nothing down from 3.
    assert registry.has_migration_path("MyVersionedObject", "1", "3")
    assert registry.has_migration_path("MyVersionedObject", "2", "1")
    assert not registry.has_migration_path("MyVersionedObject", "3", "1")
    assert registry.has_migration_path("MyVersionedObject", "2", "2")  # noop path


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
