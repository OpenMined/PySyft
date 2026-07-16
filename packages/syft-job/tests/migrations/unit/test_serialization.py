"""Versioned syft-job objects serialize with their identity and load back."""

from datetime import datetime, timezone
from pathlib import Path

from syft_migration import MigrationService

from syft_job.migrations import job_registry
from syft_job.models import JobStateV1, JobStatus, JobSubmissionMetadataV1

from .mocks import create_mock_submission, mock_submission_config_path


def test_submission_serialization(tmp_path: Path):
    path = mock_submission_config_path(tmp_path)
    submission = create_mock_submission()
    submission.save(path)

    loaded = JobSubmissionMetadataV1.load(path)
    assert loaded == submission
    assert loaded.canonical_name == "JobSubmissionMetadata"
    assert loaded.version == "1"


def test_state_serialization(tmp_path: Path):
    state = JobStateV1(
        status=JobStatus.PENDING,
        received_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
    )
    path = tmp_path / "state.yaml"
    state.save(path)

    loaded = JobStateV1.load(path)
    assert loaded == state
    assert loaded.canonical_name == "JobState"
    assert loaded.version == "1"


def test_migration_service_loads_into_versioned_class():
    service = MigrationService(registry=job_registry)

    submission = create_mock_submission()
    loaded = service.load(submission.model_dump(mode="json"))
    assert isinstance(loaded, JobSubmissionMetadataV1)

    state = JobStateV1(status=JobStatus.DONE, return_code=0)
    loaded_state = service.load(state.model_dump(mode="json"))
    assert isinstance(loaded_state, JobStateV1)
    assert loaded_state.status == JobStatus.DONE
