"""Upgrading and downgrading dataset objects across versions.

Simulates the NEXT release: V1 objects and migrations, in test scope only.
"""

import yaml
from syft_migration import MigrationRegistry, MigrationService

from syft_datasets.models import DatasetV0, PrivateDatasetConfigV0

from .mocks import create_mock_dataset, create_mock_private_config


def _register_dataset_v1(registry: MigrationRegistry) -> type:
    class DatasetV1(DatasetV0, registry=registry):
        version: str = "1"
        license: str = "unknown"  # new in v1

    registry.register_migration(
        canonical_name="Dataset",
        from_version="0",
        to_version="1",
        fn=lambda obj: DatasetV1(**obj.model_dump(exclude={"version"})),
    )
    registry.register_migration(
        canonical_name="Dataset",
        from_version="1",
        to_version="0",
        fn=lambda obj: DatasetV0(**obj.model_dump(exclude={"version", "license"})),
    )
    return DatasetV1


def _register_private_config_v1(registry: MigrationRegistry) -> type:
    class PrivateDatasetConfigV1(PrivateDatasetConfigV0, registry=registry):
        version: str = "1"
        checksum: str = ""  # new in v1

    registry.register_migration(
        canonical_name="PrivateDatasetConfig",
        from_version="0",
        to_version="1",
        fn=lambda obj: PrivateDatasetConfigV1(**obj.model_dump(exclude={"version"})),
    )
    registry.register_migration(
        canonical_name="PrivateDatasetConfig",
        from_version="1",
        to_version="0",
        fn=lambda obj: PrivateDatasetConfigV0(
            **obj.model_dump(exclude={"version", "checksum"})
        ),
    )
    return PrivateDatasetConfigV1


def _version_registry_with_migrations() -> tuple[MigrationRegistry, type, type]:
    """A fresh registry holding the current objects plus test-scope V1 versions."""
    registry = MigrationRegistry(
        protocol_name="syft-dataset",
        package_name="syft-dataset",
        package_version="test-next-release",
        protocol_version="test-next",
    )
    registry.register_object_version(DatasetV0)
    registry.register_object_version(PrivateDatasetConfigV0)
    dataset_v1 = _register_dataset_v1(registry)
    private_config_v1 = _register_private_config_v1(registry)
    return registry, dataset_v1, private_config_v1


def test_dataset_upgrades_from_disk(tmp_path):
    registry, DatasetV1, _ = _version_registry_with_migrations()
    service = MigrationService(registry=registry)

    # A current (v0) dataset file on disk loads and migrates to the next version.
    path = tmp_path / "dataset.yaml"
    path.write_text(yaml.safe_dump(create_mock_dataset().disk_dict()))
    upgraded = service.load(yaml.safe_load(path.read_text()), target_version="1")
    assert type(upgraded) is DatasetV1
    assert upgraded.name == "demo"
    assert upgraded.license == "unknown"


def test_dataset_downgrades():
    registry, DatasetV1, _ = _version_registry_with_migrations()
    service = MigrationService(registry=registry)

    v1_only = DatasetV1(
        **create_mock_dataset().model_dump(exclude={"version"}), license="mit"
    )
    downgraded = service.migrate(v1_only, target_version="0")
    assert type(downgraded) is DatasetV0
    assert downgraded.name == "demo"
    assert not hasattr(downgraded, "license")


def test_private_config_upgrades_and_downgrades():
    registry, _, PrivateDatasetConfigV1 = _version_registry_with_migrations()
    service = MigrationService(registry=registry)

    upgraded = service.migrate(create_mock_private_config(), target_version="1")
    assert type(upgraded) is PrivateDatasetConfigV1
    assert upgraded.checksum == ""

    downgraded = service.migrate(upgraded, target_version="0")
    assert type(downgraded) is PrivateDatasetConfigV0
    assert not hasattr(downgraded, "checksum")
