"""Shared mock objects for the syft-job migration tests."""

from datetime import datetime, timezone
from pathlib import Path

from syft_job.models import JobSubmissionMetadataV1

DO_EMAIL = "do@test.org"
DS_EMAIL = "ds@test.org"


def create_mock_submission() -> JobSubmissionMetadataV1:
    return JobSubmissionMetadataV1(
        name="my.job",
        submitted_by=DS_EMAIL,
        datasite_email=DO_EMAIL,
        submitted_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
        entrypoint="main.py",
        dependencies=["pandas"],
        files=["main.py"],
    )


def mock_submission_config_path(tmp_path: Path) -> Path:
    # config.yaml lives at inbox/<ds>/<job>/config.yaml under a datasite-email folder,
    # matching the path layout JobSubmissionMetadataV1.load() reverse-engineers.
    return (
        tmp_path
        / DO_EMAIL
        / "app_data"
        / "job"
        / "inbox"
        / DS_EMAIL
        / "my.job"
        / "config.yaml"
    )
