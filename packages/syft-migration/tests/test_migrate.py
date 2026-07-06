"""MigrationService moving objects between versions."""

import pytest

from syft_migration import MigrationError

from mocks import JobV1, JobV2, JobV3


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
