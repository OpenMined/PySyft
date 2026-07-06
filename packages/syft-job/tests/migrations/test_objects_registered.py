"""Every versioned syft-job object is known to the package registry."""

import importlib
import pkgutil

from syft_migration import MigratableObject

import syft_job
from syft_job.migrations import job_registry
from syft_job.models import (
    JobState,
    JobStateV1,
    JobSubmissionMetadata,
    JobSubmissionMetadataV1,
)


def test_versioned_objects_registered_and_aliased():
    # Both objects have at least one version registered in the package registry.
    assert job_registry.versions("JobSubmissionMetadata")
    assert job_registry.versions("JobState")

    # The current-version aliases resolve to the V1 classes.
    assert JobSubmissionMetadata is JobSubmissionMetadataV1
    assert JobState is JobStateV1

    # The protocol schema covers both objects and resolves a current version.
    schema = job_registry.current_protocol_schema.protocol_schema
    assert {"JobSubmissionMetadata", "JobState"} <= set(schema.supported_versions)
    assert schema.current_schema(canonical_name="JobState")
    assert schema.current_schema(canonical_name="JobSubmissionMetadata")


def _all_subclasses(cls: type) -> set[type]:
    subclasses = set(cls.__subclasses__())
    for sub in cls.__subclasses__():
        subclasses |= _all_subclasses(sub)
    return subclasses


def test_all_migratable_objects_in_package_are_registered():
    # Import every syft_job module so all MigratableObject subclasses are defined.
    for module_info in pkgutil.walk_packages(syft_job.__path__, prefix="syft_job."):
        importlib.import_module(module_info.name)

    package_objects = [
        cls
        for cls in _all_subclasses(MigratableObject)
        if cls.__module__.startswith("syft_job.")
    ]
    assert len(package_objects) >= 2  # the scan actually found the job objects

    for cls in package_objects:
        canonical_name = cls.model_fields["canonical_name"].default
        version = cls.model_fields["version"].default
        assert job_registry.get_class(canonical_name, version) is cls
