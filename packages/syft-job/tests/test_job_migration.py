"""Unit tests for the versioned syft-job objects and their migration wiring."""

import importlib
import pkgutil
from datetime import datetime, timezone
from pathlib import Path

import yaml
from syft_migration import MigratableObject, MigrationRegistry, MigrationService

import syft_job

from syft_job.migrations import job_registry
from syft_job.models import (
    JobState,
    JobStateV1,
    JobStatus,
    JobSubmissionMetadata,
    JobSubmissionMetadataV1,
)

DO_EMAIL = "do@test.org"
DS_EMAIL = "ds@test.org"


def test_versioned_objects_registered_and_aliased():
    # Both objects have at least one version registered in the package registry.
    assert job_registry.versions("JobSubmissionMetadata")
    assert job_registry.versions("JobState")

    # The current-version aliases resolve to the V1 classes.
    assert JobSubmissionMetadata is JobSubmissionMetadataV1
    assert JobState is JobStateV1

    # The protocol schema covers both objects and resolves a current version.
    schema = job_registry.current_protocol_schema
    assert {"JobSubmissionMetadata", "JobState"} <= set(schema.supported_versions)
    assert schema.current_schema(canonical_name="JobState")
    assert schema.current_schema(canonical_name="JobSubmissionMetadata")


def _all_subclasses(cls: type) -> set[type]:
    subclasses = set(cls.__subclasses__())
    for sub in cls.__subclasses__():
        subclasses |= _all_subclasses(sub)
    return subclasses


def test_all_migratable_objects_in_package_are_registered():
    # Import every syft_job module so all MigratableObject subclasses are defined.
    for module_info in pkgutil.walk_packages(syft_job.__path__, prefix="syft_job."):
        importlib.import_module(module_info.name)

    package_objects = [
        cls
        for cls in _all_subclasses(MigratableObject)
        if cls.__module__.startswith("syft_job.")
    ]
    assert len(package_objects) >= 2  # the scan actually found the job objects

    for cls in package_objects:
        canonical_name = cls.model_fields["canonical_name"].default
        version = cls.model_fields["version"].default
        assert job_registry.get_class(canonical_name, version) is cls


def create_mock_submission() -> JobSubmissionMetadataV1:
    return JobSubmissionMetadataV1(
        name="my.job",
        submitted_by=DS_EMAIL,
        datasite_email=DO_EMAIL,
        submitted_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
        entrypoint="main.py",
        dependencies=["pandas"],
        files=["main.py"],
    )


def _mock_submission_config_path(tmp_path: Path) -> Path:
    # config.yaml lives at inbox/<ds>/<job>/config.yaml under a datasite-email folder,
    # matching the path layout JobSubmissionMetadataV1.load() reverse-engineers.
    return (
        tmp_path
        / DO_EMAIL
        / "app_data"
        / "job"
        / "inbox"
        / DS_EMAIL
        / "my.job"
        / "config.yaml"
    )


def test_submission_serialization(tmp_path: Path):
    path = _mock_submission_config_path(tmp_path)
    submission = create_mock_submission()
    submission.save(path)

    loaded = JobSubmissionMetadataV1.load(path)
    assert loaded == submission
    assert loaded.canonical_name == "JobSubmissionMetadata"
    assert loaded.version == "1"


def test_state_serialization(tmp_path: Path):
    state = JobStateV1(
        status=JobStatus.PENDING,
        received_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
    )
    path = tmp_path / "state.yaml"
    state.save(path)

    loaded = JobStateV1.load(path)
    assert loaded == state
    assert loaded.canonical_name == "JobState"
    assert loaded.version == "1"


# --- simulate the NEXT release: V2 objects + migrations, in test scope only ---


def _register_state_v2(registry: MigrationRegistry) -> type:
    class JobStateV2(JobStateV1, registry=registry):
        version: str = "2"
        retries: int = 0  # new in v2

    registry.register_migration(
        canonical_name="JobState",
        from_version="1",
        to_version="2",
        fn=lambda obj: JobStateV2(**obj.model_dump(exclude={"version"})),
    )
    registry.register_migration(
        canonical_name="JobState",
        from_version="2",
        to_version="1",
        fn=lambda obj: JobStateV1(**obj.model_dump(exclude={"version", "retries"})),
    )
    return JobStateV2


def _register_submission_v2(registry: MigrationRegistry) -> type:
    class JobSubmissionMetadataV2(JobSubmissionMetadataV1, registry=registry):
        version: str = "2"
        priority: str = "normal"  # new in v2

    registry.register_migration(
        canonical_name="JobSubmissionMetadata",
        from_version="1",
        to_version="2",
        fn=lambda obj: JobSubmissionMetadataV2(**obj.model_dump(exclude={"version"})),
    )
    registry.register_migration(
        canonical_name="JobSubmissionMetadata",
        from_version="2",
        to_version="1",
        fn=lambda obj: JobSubmissionMetadataV1(
            **obj.model_dump(exclude={"version", "priority"})
        ),
    )
    return JobSubmissionMetadataV2


def _next_version_registry() -> tuple[MigrationRegistry, type, type]:
    """A fresh registry holding the current objects plus test-scope V2 versions."""
    registry = MigrationRegistry(
        protocol_name="syft-job",
        package_name="syft-job",
        package_version="test-next-release",
    )
    registry.register_object_version(JobStateV1)
    registry.register_object_version(JobSubmissionMetadataV1)
    state_v2 = _register_state_v2(registry)
    submission_v2 = _register_submission_v2(registry)
    return registry, state_v2, submission_v2


def test_job_state_upgrades_from_disk_and_downgrades(tmp_path: Path):
    registry, JobStateV2, _ = _next_version_registry()
    service = MigrationService(registry=registry)

    # An old (v1) state file on disk loads and migrates to the next version.
    path = tmp_path / "state.yaml"
    JobStateV1(status=JobStatus.APPROVED, approved_by=DO_EMAIL).save(path)
    upgraded = service.load(yaml.safe_load(path.read_text()), target_version="2")
    assert type(upgraded) is JobStateV2
    assert upgraded.status == JobStatus.APPROVED
    assert upgraded.approved_by == DO_EMAIL
    assert upgraded.retries == 0

    # A v2-only object in memory downgrades to the old version.
    downgraded = service.migrate(
        JobStateV2(status=JobStatus.DONE, retries=2), target_version="1"
    )
    assert type(downgraded) is JobStateV1
    assert downgraded.status == JobStatus.DONE
    assert not hasattr(downgraded, "retries")


def test_job_submission_upgrades_from_disk_and_downgrades(tmp_path: Path):
    registry, _, JobSubmissionMetadataV2 = _next_version_registry()
    service = MigrationService(registry=registry)

    # An old (v1) config file on disk loads and migrates to the next version.
    path = _mock_submission_config_path(tmp_path)
    create_mock_submission().save(path)
    upgraded = service.migrate(JobSubmissionMetadataV1.load(path), target_version="2")
    assert type(upgraded) is JobSubmissionMetadataV2
    assert upgraded.name == "my.job"
    assert upgraded.priority == "normal"

    # A v2-only object in memory downgrades to the old version.
    downgraded = service.migrate(upgraded, target_version="1")
    assert type(downgraded) is JobSubmissionMetadataV1
    assert downgraded.name == "my.job"
    assert not hasattr(downgraded, "priority")


def test_migration_service_loads_into_versioned_class():
    service = MigrationService(registry=job_registry)

    submission = create_mock_submission()
    loaded = service.load(submission.model_dump(mode="json"))
    assert isinstance(loaded, JobSubmissionMetadataV1)

    state = JobStateV1(status=JobStatus.DONE, return_code=0)
    loaded_state = service.load(state.model_dump(mode="json"))
    assert isinstance(loaded_state, JobStateV1)
    assert loaded_state.status == JobStatus.DONE
