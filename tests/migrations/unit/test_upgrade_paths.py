"""Every registered object version can migrate up to latest and down to any lower."""

from syft.migrations.registry import client_registry
from syft_migration import missing_downgrade_paths, missing_upgrade_paths


def test_every_version_has_upgrade_path_to_latest():
    assert client_registry.objects  # sanity: the registry is populated
    assert missing_upgrade_paths(client_registry) == []


def test_every_version_has_downgrade_path_to_all_lower_versions():
    assert client_registry.objects  # sanity: the registry is populated
    assert missing_downgrade_paths(client_registry) == []
