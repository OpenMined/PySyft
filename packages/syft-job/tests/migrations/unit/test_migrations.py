"""Migrations against the REAL job registry, using on-disk fixtures.

Every registered version of each object has a serialized fixture under
``fixtures/<canonical_name>/v<version>.yaml``; each fixture loads and upgrades
in memory to the latest version, and the latest version downgrades to every
registered version and writes to disk. With only V1 registered these are
no-ops; when a newer version is registered, these tests fail until a fixture
for it is added and then exercise the real migration paths automatically.
"""

from pathlib import Path

import yaml
from syft_migration import MigrationService

from syft_job.migrations import job_registry
from syft_job.models import JobStatus

from .mocks import mock_submission_config_path

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(canonical_name: str, version: str) -> dict:
    path = FIXTURES_DIR / canonical_name / f"v{version}.yaml"
    assert path.exists(), (
        f"Missing fixture {path}; add one for every registered version"
    )
    return yaml.safe_load(path.read_text())


def test_all_job_state_versions_upgrade_from_disk_to_latest():
    service = MigrationService(registry=job_registry)
    latest = job_registry.latest_version("JobState")
    latest_cls = job_registry.get_class("JobState", latest)

    for version in job_registry.versions("JobState"):
        upgraded = service.load(
            _load_fixture("JobState", version), target_version=latest
        )
        assert type(upgraded) is latest_cls
        assert upgraded.status == JobStatus.DONE


def test_latest_job_state_downgrades_to_all_versions(tmp_path: Path):
    service = MigrationService(registry=job_registry)
    latest = job_registry.latest_version("JobState")
    newest = service.load(_load_fixture("JobState", latest))

    for version in job_registry.versions("JobState"):
        version_cls = job_registry.get_class("JobState", version)
        downgraded = service.migrate(newest, target_version=version)
        assert type(downgraded) is version_cls

        path = tmp_path / f"state_v{version}.yaml"
        downgraded.save(path)
        assert version_cls.load(path) == downgraded


def test_all_submission_versions_upgrade_from_disk_to_latest():
    service = MigrationService(registry=job_registry)
    latest = job_registry.latest_version("JobSubmissionMetadata")
    latest_cls = job_registry.get_class("JobSubmissionMetadata", latest)

    for version in job_registry.versions("JobSubmissionMetadata"):
        upgraded = service.load(
            _load_fixture("JobSubmissionMetadata", version), target_version=latest
        )
        assert type(upgraded) is latest_cls
        assert upgraded.name == "my.job"


def test_latest_submission_downgrades_to_all_versions(tmp_path: Path):
    service = MigrationService(registry=job_registry)
    latest = job_registry.latest_version("JobSubmissionMetadata")
    newest = service.load(_load_fixture("JobSubmissionMetadata", latest))

    for version in job_registry.versions("JobSubmissionMetadata"):
        version_cls = job_registry.get_class("JobSubmissionMetadata", version)
        downgraded = service.migrate(newest, target_version=version)
        assert type(downgraded) is version_cls

        path = mock_submission_config_path(tmp_path / f"v{version}")
        downgraded.save(path)
        assert version_cls.load(path) == downgraded
