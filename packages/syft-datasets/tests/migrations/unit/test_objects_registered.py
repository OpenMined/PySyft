"""Every versioned syft-dataset object is known to the package registry."""

import importlib
import pkgutil

from syft_migration import MigratableObject

import syft_datasets
from syft_datasets.migrations import dataset_registry
from syft_datasets.models import (
    Dataset,
    DatasetV1,
    PrivateDatasetConfig,
    PrivateDatasetConfigV1,
)


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


def _all_subclasses(cls: type) -> set[type]:
    subclasses = set(cls.__subclasses__())
    for sub in cls.__subclasses__():
        subclasses |= _all_subclasses(sub)
    return subclasses


def test_all_migratable_objects_in_package_are_registered():
    # Import every syft_datasets module so all MigratableObject subclasses are defined.
    for module_info in pkgutil.walk_packages(
        syft_datasets.__path__, prefix="syft_datasets."
    ):
        importlib.import_module(module_info.name)

    package_objects = [
        cls
        for cls in _all_subclasses(MigratableObject)
        if cls.__module__.startswith("syft_datasets.")
    ]
    assert len(package_objects) >= 2  # the scan actually found the dataset objects

    for cls in package_objects:
        canonical_name = cls.model_fields["canonical_name"].default
        version = cls.model_fields["version"].default
        assert dataset_registry.get_class(canonical_name, version) is cls
