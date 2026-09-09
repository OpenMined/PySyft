"""Migrations against the REAL dataset registry, using on-disk fixtures.

Every registered version of each object has a serialized fixture under
``fixtures/<canonical_name>/v<version>.yaml``; each fixture loads and upgrades
in memory to the latest version, and the latest version downgrades to every
registered version and round-trips through disk. With only V1 registered these
are no-ops; when a newer version is registered, these tests fail until a fixture
for it is added and then exercise the real migration paths automatically.
"""

from pathlib import Path

import yaml
from syft_migration import MigrationService

from syft_datasets.migrations import dataset_registry

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(canonical_name: str, version: str) -> dict:
    path = FIXTURES_DIR / canonical_name / f"v{version}.yaml"
    assert path.exists(), (
        f"Missing fixture {path}; add one for every registered version"
    )
    return yaml.safe_load(path.read_text())


def _assert_all_upgrade_to_latest(canonical_name: str):
    service = MigrationService(registry=dataset_registry)
    latest = dataset_registry.latest_version(canonical_name)
    latest_cls = dataset_registry.get_class(canonical_name, latest)

    for version in dataset_registry.versions(canonical_name):
        upgraded = service.load(
            _load_fixture(canonical_name, version), target_version=latest
        )
        assert type(upgraded) is latest_cls


def _assert_latest_downgrades_to_all(canonical_name: str, tmp_path: Path):
    service = MigrationService(registry=dataset_registry)
    latest = dataset_registry.latest_version(canonical_name)
    newest = service.load(_load_fixture(canonical_name, latest))

    for version in dataset_registry.versions(canonical_name):
        version_cls = dataset_registry.get_class(canonical_name, version)
        downgraded = service.migrate(newest, target_version=version)
        assert type(downgraded) is version_cls

        path = tmp_path / f"{canonical_name}_v{version}.yaml"
        path.write_text(yaml.safe_dump(downgraded.disk_dict()))
        reloaded = service.load(
            yaml.safe_load(path.read_text()), target_version=version
        )
        assert reloaded == downgraded


def test_all_dataset_versions_upgrade_from_disk_to_latest():
    _assert_all_upgrade_to_latest("Dataset")


def test_latest_dataset_downgrades_to_all_versions(tmp_path: Path):
    _assert_latest_downgrades_to_all("Dataset", tmp_path)


def test_all_private_config_versions_upgrade_from_disk_to_latest():
    _assert_all_upgrade_to_latest("PrivateDatasetConfig")


def test_latest_private_config_downgrades_to_all_versions(tmp_path: Path):
    _assert_latest_downgrades_to_all("PrivateDatasetConfig", tmp_path)
