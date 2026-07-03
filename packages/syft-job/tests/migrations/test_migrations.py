"""Upgrading and downgrading job objects across versions.

Simulates the NEXT release: V2 objects and migrations, in test scope only.
"""

from pathlib import Path

import yaml
from syft_migration import MigrationRegistry, MigrationService

from syft_job.models import JobStateV1, JobStatus, JobSubmissionMetadataV1

from .mocks import DO_EMAIL, create_mock_submission, mock_submission_config_path


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


def _version_registry_with_migrations() -> tuple[MigrationRegistry, type, type]:
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
    registry, JobStateV2, _ = _version_registry_with_migrations()
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
    registry, _, JobSubmissionMetadataV2 = _version_registry_with_migrations()
    service = MigrationService(registry=registry)

    # An old (v1) config file on disk loads and migrates to the next version.
    path = mock_submission_config_path(tmp_path)
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
