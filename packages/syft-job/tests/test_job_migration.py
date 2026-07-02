"""Unit tests for the versioned syft-job objects and their migration wiring."""

from datetime import datetime, timezone
from pathlib import Path

from syft_migration import MigrationService

from syft_job.models import (
    JobState,
    JobStateV1,
    JobStatus,
    JobSubmissionMetadata,
    JobSubmissionMetadataV1,
    job_registry,
)

DO_EMAIL = "do@test.org"
DS_EMAIL = "ds@test.org"


def test_versioned_objects_registered_and_aliased():
    # Both objects register their single version into the package registry.
    assert job_registry.versions("job_submission_metadata") == ["1"]
    assert job_registry.versions("job_state") == ["1"]

    # The current-version aliases resolve to the V1 classes.
    assert JobSubmissionMetadata is JobSubmissionMetadataV1
    assert JobState is JobStateV1

    # The protocol schema pins both objects to version "1".
    schema = job_registry.current_protocol_schema
    assert schema.object_versions == {
        "job_submission_metadata": "1",
        "job_state": "1",
    }


def _make_submission() -> JobSubmissionMetadataV1:
    return JobSubmissionMetadataV1(
        name="my.job",
        submitted_by=DS_EMAIL,
        datasite_email=DO_EMAIL,
        submitted_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
        entrypoint="main.py",
        dependencies=["pandas"],
        files=["main.py"],
    )


def test_submission_round_trip_carries_identity(tmp_path: Path):
    # config.yaml lives at inbox/<ds>/<job>/config.yaml under a datasite-email folder,
    # matching the path layout JobSubmissionMetadataV1.load() reverse-engineers.
    path = (
        tmp_path
        / DO_EMAIL
        / "app_data"
        / "job"
        / "inbox"
        / DS_EMAIL
        / "my.job"
        / "config.yaml"
    )
    submission = _make_submission()
    submission.save(path)

    loaded = JobSubmissionMetadataV1.load(path)
    assert loaded == submission
    assert loaded.canonical_name == "job_submission_metadata"
    assert loaded.version == "1"


def test_state_round_trip_carries_identity(tmp_path: Path):
    state = JobStateV1(
        status=JobStatus.PENDING,
        received_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
    )
    path = tmp_path / "state.yaml"
    state.save(path)

    loaded = JobStateV1.load(path)
    assert loaded == state
    assert loaded.canonical_name == "job_state"
    assert loaded.version == "1"


def test_migration_service_loads_into_versioned_class():
    service = MigrationService(registry=job_registry)

    submission = _make_submission()
    loaded = service.load(submission.model_dump(mode="json"))
    assert isinstance(loaded, JobSubmissionMetadataV1)

    state = JobStateV1(status=JobStatus.DONE, return_code=0)
    loaded_state = service.load(state.model_dump(mode="json"))
    assert isinstance(loaded_state, JobStateV1)
    assert loaded_state.status == JobStatus.DONE
