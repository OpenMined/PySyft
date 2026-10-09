"""Every versioned syft-dataset object is known to the package registry."""

import syft_datasets
from syft_datasets.migrations import dataset_registry
from syft_datasets.models import (
    Dataset,
    DatasetV1,
    PrivateDatasetConfig,
    PrivateDatasetConfigV1,
)
from syft_migration import unregistered_objects, versioned_objects


def test_versioned_objects_registered_and_aliased():
    # Both objects have at least one version registered in the package registry.
    assert dataset_registry.versions("Dataset")
    assert dataset_registry.versions("PrivateDatasetConfig")

    # The current-version aliases resolve to the V1 classes.
    assert Dataset is DatasetV1
    assert PrivateDatasetConfig is PrivateDatasetConfigV1

    # The protocol schema covers both objects and resolves a current version.
    schema = dataset_registry.compute_protocol_schema()
    assert {"Dataset", "PrivateDatasetConfig"} <= set(schema.supported_versions)
    assert schema.current_schema(canonical_name="Dataset")
    assert schema.current_schema(canonical_name="PrivateDatasetConfig")


def test_all_migratable_objects_in_package_are_registered():
    # The scan imports every syft_datasets module, so it sees objects that
    # nothing else imports.
    assert len(versioned_objects(syft_datasets)) >= 2
    assert unregistered_objects(dataset_registry, syft_datasets) == []
