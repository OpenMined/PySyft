"""Every versioned syft-job object is known to the package registry."""

import syft_job
from syft_job.migrations import job_registry
from syft_job.models import (
    JobState,
    JobStateV1,
    JobSubmissionMetadata,
    JobSubmissionMetadataV1,
)
from syft_migration import unregistered_objects, versioned_objects


def test_versioned_objects_registered_and_aliased():
    # Both objects have at least one version registered in the package registry.
    assert job_registry.versions("JobSubmissionMetadata")
    assert job_registry.versions("JobState")

    # The current-version aliases resolve to the V1 classes.
    assert JobSubmissionMetadata is JobSubmissionMetadataV1
    assert JobState is JobStateV1

    # The protocol schema covers both objects and resolves a current version.
    schema = job_registry.compute_protocol_schema()
    assert {"JobSubmissionMetadata", "JobState"} <= set(schema.supported_versions)
    assert schema.current_schema(canonical_name="JobState")
    assert schema.current_schema(canonical_name="JobSubmissionMetadata")


def test_all_migratable_objects_in_package_are_registered():
    # The scan imports every syft_job module, so it sees objects that nothing
    # else imports.
    assert len(versioned_objects(syft_job)) >= 2
    assert unregistered_objects(job_registry, syft_job) == []
