"""Versioned syft-dataset objects serialize with their identity and load back."""

import yaml
from syft_migration import MigrationService

from syft_datasets.migrations import dataset_registry
from syft_datasets.models import DatasetV0, PrivateDatasetConfigV0

from .mocks import create_mock_dataset, create_mock_private_config


def test_dataset_serialization_roundtrip():
    dataset = create_mock_dataset()
    data = yaml.safe_load(yaml.safe_dump(dataset.disk_dict()))
    loaded = DatasetV0(**data)
    assert loaded == dataset
    assert loaded.canonical_name == "Dataset"
    assert loaded.version == "0"


def test_private_config_serialization_roundtrip():
    config = create_mock_private_config()
    data = yaml.safe_load(yaml.safe_dump(config.disk_dict()))
    loaded = PrivateDatasetConfigV0(**data)
    assert loaded == config
    assert loaded.canonical_name == "PrivateDatasetConfig"
    assert loaded.version == "0"


def test_migration_service_loads_into_versioned_class():
    service = MigrationService(registry=dataset_registry)

    dataset = create_mock_dataset()
    loaded = service.load(dataset.disk_dict())
    assert isinstance(loaded, DatasetV0)
    assert loaded.name == "demo"

    config = create_mock_private_config()
    loaded_config = service.load(config.disk_dict())
    assert isinstance(loaded_config, PrivateDatasetConfigV0)
