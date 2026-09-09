"""Upgrading and downgrading dataset objects across versions.

Simulates the NEXT release: V2 objects and migrations, in test scope only.
"""

import yaml
from syft_migration import MigrationRegistry, MigrationService

from syft_datasets.models import DatasetV1, PrivateDatasetConfigV1

from .mocks import create_mock_dataset, create_mock_private_config


def _register_dataset_v2(registry: MigrationRegistry) -> type:
    class DatasetV2(DatasetV1, registry=registry):
        version: str = "2"
        license: str = "unknown"  # new in v2

    registry.register_migration(
        canonical_name="Dataset",
        from_version="1",
        to_version="2",
        fn=lambda obj: DatasetV2(**obj.model_dump(exclude={"version"})),
    )
    registry.register_migration(
        canonical_name="Dataset",
        from_version="2",
        to_version="1",
        fn=lambda obj: DatasetV1(**obj.model_dump(exclude={"version", "license"})),
    )
    return DatasetV2


def _register_private_config_v2(registry: MigrationRegistry) -> type:
    class PrivateDatasetConfigV2(PrivateDatasetConfigV1, registry=registry):
        version: str = "2"
        checksum: str = ""  # new in v2

    registry.register_migration(
        canonical_name="PrivateDatasetConfig",
        from_version="1",
        to_version="2",
        fn=lambda obj: PrivateDatasetConfigV2(**obj.model_dump(exclude={"version"})),
    )
    registry.register_migration(
        canonical_name="PrivateDatasetConfig",
        from_version="2",
        to_version="1",
        fn=lambda obj: PrivateDatasetConfigV1(
            **obj.model_dump(exclude={"version", "checksum"})
        ),
    )
    return PrivateDatasetConfigV2


def _version_registry_with_migrations() -> tuple[MigrationRegistry, type, type]:
    """A fresh registry holding the current objects plus test-scope V2 versions."""
    registry = MigrationRegistry(
        protocol_name="syft-dataset",
        package_name="syft-dataset",
        package_version="test-next-release",
        protocol_version="test-next",
    )
    registry.register_object_version(DatasetV1)
    registry.register_object_version(PrivateDatasetConfigV1)
    dataset_v2 = _register_dataset_v2(registry)
    private_config_v2 = _register_private_config_v2(registry)
    return registry, dataset_v2, private_config_v2


def test_dataset_upgrades_from_disk(tmp_path):
    registry, DatasetV2, _ = _version_registry_with_migrations()
    service = MigrationService(registry=registry)

    # An old (v1) dataset file on disk loads and migrates to the next version.
    path = tmp_path / "dataset.yaml"
    path.write_text(yaml.safe_dump(create_mock_dataset().disk_dict()))
    upgraded = service.load(yaml.safe_load(path.read_text()), target_version="2")
    assert type(upgraded) is DatasetV2
    assert upgraded.name == "demo"
    assert upgraded.license == "unknown"


def test_dataset_downgrades():
    registry, DatasetV2, _ = _version_registry_with_migrations()
    service = MigrationService(registry=registry)

    v2_only = DatasetV2(
        **create_mock_dataset().model_dump(exclude={"version"}), license="mit"
    )
    downgraded = service.migrate(v2_only, target_version="1")
    assert type(downgraded) is DatasetV1
    assert downgraded.name == "demo"
    assert not hasattr(downgraded, "license")


def test_private_config_upgrades_and_downgrades():
    registry, _, PrivateDatasetConfigV2 = _version_registry_with_migrations()
    service = MigrationService(registry=registry)

    upgraded = service.migrate(create_mock_private_config(), target_version="2")
    assert type(upgraded) is PrivateDatasetConfigV2
    assert upgraded.checksum == ""

    downgraded = service.migrate(upgraded, target_version="1")
    assert type(downgraded) is PrivateDatasetConfigV1
    assert not hasattr(downgraded, "checksum")
