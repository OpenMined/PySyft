"""MigrationService moving objects between versions."""

import pytest

from syft_migration import MigrationError

from mocks import MyVersionedObjectV1, MyVersionedObjectV2, MyVersionedObjectV3


def test_migrate_upgrades_single_step(service):
    result = service.migrate(obj=MyVersionedObjectV1(name="x"), target_version="2")
    assert isinstance(result, MyVersionedObjectV2)
    assert result.name == "x" and result.owner == "unknown"


def test_migrate_chains_multiple_steps(service):
    result = service.migrate(obj=MyVersionedObjectV1(name="x"), target_version="3")
    assert isinstance(result, MyVersionedObjectV3)
    assert result.version == "3"


def test_migrate_downgrades(service):
    result = service.migrate(
        obj=MyVersionedObjectV2(name="x", owner="bob"), target_version="1"
    )
    assert isinstance(result, MyVersionedObjectV1)
    assert result.name == "x"


def test_migrate_same_version_is_noop(service):
    obj = MyVersionedObjectV2(name="x")
    assert service.migrate(obj=obj, target_version="2") is obj


def test_migrate_missing_path_raises(service):
    with pytest.raises(MigrationError):
        service.migrate(obj=MyVersionedObjectV3(name="x"), target_version="1")


def test_downgrade_for_protocol_version(service):
    result = service.downgrade_for_protocol_version(
        obj=MyVersionedObjectV2(name="x", owner="bob"), protocol_version="1"
    )
    assert isinstance(result, MyVersionedObjectV1)
    assert result.name == "x"


def test_load_deserializes_into_historical_class(service):
    obj = service.load(
        data={"canonical_name": "MyVersionedObject", "version": "1", "name": "x"}
    )
    assert isinstance(obj, MyVersionedObjectV1)
    migrated = service.migrate(
        obj=obj,
        target_version=service.registry.latest_version(
            canonical_name="MyVersionedObject"
        ),
    )
    assert isinstance(migrated, MyVersionedObjectV3)


def test_load_unknown_object_raises(service):
    with pytest.raises(MigrationError):
        service.load(data={"canonical_name": "MyVersionedObject", "version": "99"})
