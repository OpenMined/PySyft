"""The hardcoded release artifacts of past syft-job releases."""

from syft_migration import MigrationService, ReleaseArtifact

from syft_job.migrations import job_registry
from syft_job.migrations.history import HISTORY_DIR
from syft_job.models import JobStateV1, JobStatus


def test_0_1_38_artifact_file_loads():
    artifact = ReleaseArtifact.load(HISTORY_DIR / "syft-job-0.1.38.json")
    assert artifact.protocol_schema.protocol_name == "syft-job"
    assert artifact.protocol_schema.version == "0"
    assert artifact.package_schema.package_version == "0.1.38"
    assert artifact.package_schema.protocol_schema.version == "0"
    expected_versions = {"JobState": ["1"], "JobSubmissionMetadata": ["1"]}
    assert artifact.protocol_schema.supported_versions == expected_versions
    assert artifact.package_schema.protocol_schema.supported_versions == (
        expected_versions
    )


def test_historic_schemas_registered_on_import():
    # syft_job/__init__ registers every artifact in migrations/history/.
    assert "0.1.38" in job_registry.history_protocol_schemas
    schema = job_registry.schema_for_protocol_version("0")
    assert schema.current_schema("JobState") == "1"
    assert schema.current_schema("JobSubmissionMetadata") == "1"


def test_downgrade_for_last_released_package_version():
    service = MigrationService(registry=job_registry)
    state = JobStateV1(status=JobStatus.DONE)
    downgraded = service.downgrade_for_package_version(state, "0.1.38")
    assert isinstance(downgraded, JobStateV1)
    assert downgraded.status == JobStatus.DONE
