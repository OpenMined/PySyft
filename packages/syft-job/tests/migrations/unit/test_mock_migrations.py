"""Upgrading and downgrading job objects across versions.

Simulates the NEXT release: V1 objects and migrations, in test scope only.
"""

from pathlib import Path

import yaml
from syft_migration import MigrationRegistry, MigrationService

from syft_job.models import JobStateV0, JobStatus, JobSubmissionMetadataV0

from .mocks import DO_EMAIL, create_mock_submission, mock_submission_config_path


def _register_state_v1(registry: MigrationRegistry) -> type:
    class JobStateV1(JobStateV0, registry=registry):
        version: str = "1"
        retries: int = 0  # new in v1

    registry.register_migration(
        canonical_name="JobState",
        from_version="0",
        to_version="1",
        fn=lambda obj: JobStateV1(**obj.model_dump(exclude={"version"})),
    )
    registry.register_migration(
        canonical_name="JobState",
        from_version="1",
        to_version="0",
        fn=lambda obj: JobStateV0(**obj.model_dump(exclude={"version", "retries"})),
    )
    return JobStateV1


def _register_submission_v1(registry: MigrationRegistry) -> type:
    class JobSubmissionMetadataV1(JobSubmissionMetadataV0, registry=registry):
        version: str = "1"
        priority: str = "normal"  # new in v1

    registry.register_migration(
        canonical_name="JobSubmissionMetadata",
        from_version="0",
        to_version="1",
        fn=lambda obj: JobSubmissionMetadataV1(**obj.model_dump(exclude={"version"})),
    )
    registry.register_migration(
        canonical_name="JobSubmissionMetadata",
        from_version="1",
        to_version="0",
        fn=lambda obj: JobSubmissionMetadataV0(
            **obj.model_dump(exclude={"version", "priority"})
        ),
    )
    return JobSubmissionMetadataV1


def _version_registry_with_migrations() -> tuple[MigrationRegistry, type, type]:
    """A fresh registry holding the current objects plus test-scope V1 versions."""
    registry = MigrationRegistry(
        protocol_name="syft-job",
        package_name="syft-job",
        package_version="test-next-release",
        protocol_version="test-next",
    )
    registry.register_object_version(JobStateV0)
    registry.register_object_version(JobSubmissionMetadataV0)
    state_v1 = _register_state_v1(registry)
    submission_v1 = _register_submission_v1(registry)
    return registry, state_v1, submission_v1


def test_job_state_upgrades_from_disk(tmp_path: Path):
    registry, JobStateV1, _ = _version_registry_with_migrations()
    service = MigrationService(registry=registry)

    # A current (v0) state file on disk loads and migrates to the next version.
    path = tmp_path / "state.yaml"
    JobStateV0(status=JobStatus.APPROVED, approved_by=DO_EMAIL).save(path)
    upgraded = service.load(yaml.safe_load(path.read_text()), target_version="1")
    assert type(upgraded) is JobStateV1
    assert upgraded.status == JobStatus.APPROVED
    assert upgraded.approved_by == DO_EMAIL
    assert upgraded.retries == 0


def test_job_state_downgrades():
    registry, JobStateV1, _ = _version_registry_with_migrations()
    service = MigrationService(registry=registry)

    # A v1-only object in memory downgrades to the current version.
    downgraded = service.migrate(
        JobStateV1(status=JobStatus.DONE, retries=2), target_version="0"
    )
    assert type(downgraded) is JobStateV0
    assert downgraded.status == JobStatus.DONE
    assert not hasattr(downgraded, "retries")


def test_job_submission_upgrades_from_disk(tmp_path: Path):
    registry, _, JobSubmissionMetadataV1 = _version_registry_with_migrations()
    service = MigrationService(registry=registry)

    # A current (v0) config file on disk loads and migrates to the next version.
    path = mock_submission_config_path(tmp_path)
    create_mock_submission().save(path)
    upgraded = service.migrate(JobSubmissionMetadataV0.load(path), target_version="1")
    assert type(upgraded) is JobSubmissionMetadataV1
    assert upgraded.name == "my.job"
    assert upgraded.priority == "normal"


def test_job_submission_downgrades():
    registry, _, JobSubmissionMetadataV1 = _version_registry_with_migrations()
    service = MigrationService(registry=registry)

    # A v1-only object in memory downgrades to the current version.
    v1_only = JobSubmissionMetadataV1(
        **create_mock_submission().model_dump(exclude={"version"}), priority="high"
    )
    downgraded = service.migrate(v1_only, target_version="0")
    assert type(downgraded) is JobSubmissionMetadataV0
    assert downgraded.name == "my.job"
    assert not hasattr(downgraded, "priority")
