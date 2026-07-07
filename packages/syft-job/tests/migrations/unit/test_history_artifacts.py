"""The hardcoded release artifacts of past syft-job releases."""

from syft_migration import (
    MigrationService,
    ReleasedPackageProtocolInfo,
    ReleasedProtocol,
)

from syft_job.migrations import job_registry
from syft_job.migrations.history import PACKAGE_ARTIFACTS_DIR, PROTOCOLS_DIR
from syft_job.models import JobStateV1, JobStatus


def test_0_1_38_artifact_file_loads():
    artifact = ReleasedPackageProtocolInfo.load(
        PACKAGE_ARTIFACTS_DIR / "syft-job-0.1.38.json"
    )
    assert artifact.package_info.package_name == "syft-job"
    assert artifact.package_info.version == "0.1.38"
    assert artifact.package_info.protocol_version == "0"
    assert artifact.protocol_schema.protocol_name == "syft-job"
    assert artifact.protocol_schema.version == "0"
    expected_versions = {"JobState": ["1"], "JobSubmissionMetadata": ["1"]}
    assert artifact.protocol_schema.supported_versions == expected_versions


def test_protocol_0_released_protocol_loads():
    released = ReleasedProtocol.load(PROTOCOLS_DIR / "protocol-0.json")
    schema = released.protocol_schema
    assert schema.version == "0"
    assert set(schema.current_object_schemas) == {
        "JobState",
        "JobSubmissionMetadata",
    }
    assert schema.current_schema("JobState") == "1"
    assert schema.current_schema("JobSubmissionMetadata") == "1"


def test_released_object_schemas_unchanged():
    # Released object versions are frozen forever; see the failure message.
    drift = job_registry.find_schema_drift()
    assert drift == [], (
        f"Released object schemas changed: {drift} "
        "(canonical_name, object_version, protocol_version).\n"
        "A class that shipped in a released protocol was modified in place. "
        "Released versions are frozen forever, because peers on old releases "
        "still read/write them. To fix:\n"
        "  1. Revert your change to the released class (e.g. JobStateV1).\n"
        "  2. Create the next version instead: copy the class into "
        "models/<object>/v<x+1>.py with version='<x+1>' and apply your change "
        "there (copy changed nested objects like enums into the new file too).\n"
        "  3. Point the current-version alias in models/<object>/__init__.py "
        "at the new class.\n"
        "  4. Register migrations in BOTH directions (v<x> -> v<x+1> and back) "
        "so old peers stay supported; test_upgrade_paths enforces this.\n"
        "  5. Add a serialized fixture: tests/migrations/unit/fixtures/"
        "<CanonicalName>/v<x+1>.yaml.\n"
        "  6. Bump JOB_PROTOCOL_VERSION in syft_job/migrations/registry.py — a "
        "new object version is a protocol change.\n"
        "If NO class was edited and this failure appeared after a pydantic "
        "upgrade, model_json_schema() output changed cosmetically: review the "
        "diff carefully and regenerate the files under migrations/history/."
    )


def test_protocol_bumped_when_changed():
    # Fires when the job protocol changes (e.g. a new object version registers)
    # without bumping JOB_PROTOCOL_VERSION.
    assert not job_registry.protocol_changed_without_bump()


def test_historic_schemas_registered_on_import():
    # syft_job/__init__ registers every artifact in migrations/history/.
    assert job_registry.package_version_history["0"].version == "0.1.38"
    schema = job_registry.schema_for_protocol_version("0")
    assert schema.current_schema("JobState") == "1"
    assert schema.current_schema("JobSubmissionMetadata") == "1"


def test_downgrade_for_last_released_protocol_version():
    service = MigrationService(registry=job_registry)
    state = JobStateV1(status=JobStatus.DONE)
    downgraded = service.downgrade_for_protocol_version(state, "0")
    assert isinstance(downgraded, JobStateV1)
    assert downgraded.status == JobStatus.DONE
