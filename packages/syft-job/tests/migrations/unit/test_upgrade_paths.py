"""Every registered object version can migrate up to latest and down to any lower."""

from syft_job.migrations import job_registry
from syft_migration import missing_downgrade_paths, missing_upgrade_paths


def test_every_version_has_upgrade_path_to_latest():
    assert job_registry.objects  # sanity: the registry is populated
    assert missing_upgrade_paths(job_registry) == []


def test_every_version_has_downgrade_path_to_all_lower_versions():
    assert job_registry.objects  # sanity: the registry is populated
    assert missing_downgrade_paths(job_registry) == []
