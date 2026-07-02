from importlib.metadata import version

from syft_migration import PackageProtocolSchema

from .job_state_v1 import JobStateV1
from .job_submission_metadata_v1 import JobSubmissionMetadataV1
from .migration import job_registry

# Pins the object versions that this release of syft-job ships.
schema = PackageProtocolSchema.from_objects(
    protocol_name="syft-job",
    package_name="syft-job",
    package_version=version("syft-job"),
    classes=[JobSubmissionMetadataV1, JobStateV1],
)
job_registry.register_protocol_schema(schema=schema, current=True)
