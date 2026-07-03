"""Migrations against the REAL job registry.

Every registered version of each object loads from disk and upgrades in memory
to the latest version, and the latest version downgrades to every registered
version and writes to disk. With only V1 registered these are no-ops; the
moment a newer version is registered, these tests exercise the real migration
paths automatically.
"""

from pathlib import Path

from syft_migration import MigratableObject, MigrationService

from syft_job.migrations import job_registry
from syft_job.models import JobState, JobStatus

from .mocks import create_mock_submission, mock_submission_config_path


def _instance_of(version_cls: type, reference: MigratableObject) -> MigratableObject:
    """Build an instance of ``version_cls`` from a current-version reference object.

    Copies the reference fields that ``version_cls`` knows; fails loudly if a
    version requires a field the reference lacks (update the mocks then).
    """
    data = reference.model_dump(exclude={"canonical_name", "version"})
    return version_cls(
        **{key: value for key, value in data.items() if key in version_cls.model_fields}
    )


def _reference_state() -> JobState:
    return JobState(status=JobStatus.DONE, return_code=0)


def test_all_job_state_versions_upgrade_from_disk_to_latest(tmp_path: Path):
    service = MigrationService(registry=job_registry)
    latest = job_registry.latest_version("JobState")
    latest_cls = job_registry.get_class("JobState", latest)

    for version in job_registry.versions("JobState"):
        version_cls = job_registry.get_class("JobState", version)
        path = tmp_path / f"state_v{version}.yaml"
        _instance_of(version_cls, _reference_state()).save(path)

        upgraded = service.migrate(version_cls.load(path), target_version=latest)
        assert type(upgraded) is latest_cls
        assert upgraded.status == JobStatus.DONE


def test_latest_job_state_downgrades_to_all_versions(tmp_path: Path):
    service = MigrationService(registry=job_registry)
    latest = job_registry.latest_version("JobState")
    newest = _instance_of(job_registry.get_class("JobState", latest), _reference_state())

    for version in job_registry.versions("JobState"):
        version_cls = job_registry.get_class("JobState", version)
        downgraded = service.migrate(newest, target_version=version)
        assert type(downgraded) is version_cls

        path = tmp_path / f"state_v{version}.yaml"
        downgraded.save(path)
        assert version_cls.load(path) == downgraded


def test_all_submission_versions_upgrade_from_disk_to_latest(tmp_path: Path):
    service = MigrationService(registry=job_registry)
    latest = job_registry.latest_version("JobSubmissionMetadata")
    latest_cls = job_registry.get_class("JobSubmissionMetadata", latest)

    for version in job_registry.versions("JobSubmissionMetadata"):
        version_cls = job_registry.get_class("JobSubmissionMetadata", version)
        path = mock_submission_config_path(tmp_path / f"v{version}")
        _instance_of(version_cls, create_mock_submission()).save(path)

        upgraded = service.migrate(version_cls.load(path), target_version=latest)
        assert type(upgraded) is latest_cls
        assert upgraded.name == "my.job"


def test_latest_submission_downgrades_to_all_versions(tmp_path: Path):
    service = MigrationService(registry=job_registry)
    latest = job_registry.latest_version("JobSubmissionMetadata")
    newest = _instance_of(
        job_registry.get_class("JobSubmissionMetadata", latest),
        create_mock_submission(),
    )

    for version in job_registry.versions("JobSubmissionMetadata"):
        version_cls = job_registry.get_class("JobSubmissionMetadata", version)
        downgraded = service.migrate(newest, target_version=version)
        assert type(downgraded) is version_cls

        path = mock_submission_config_path(tmp_path / f"v{version}")
        downgraded.save(path)
        assert version_cls.load(path) == downgraded
