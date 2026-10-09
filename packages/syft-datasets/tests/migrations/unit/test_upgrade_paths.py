"""Every registered object version can migrate up to latest and down to any lower."""

from syft_datasets.migrations import dataset_registry
from syft_migration import missing_downgrade_paths, missing_upgrade_paths


def test_every_version_has_upgrade_path_to_latest():
    assert dataset_registry.objects  # sanity: the registry is populated
    assert missing_upgrade_paths(dataset_registry) == []


def test_every_version_has_downgrade_path_to_all_lower_versions():
    assert dataset_registry.objects  # sanity: the registry is populated
    assert missing_downgrade_paths(dataset_registry) == []
