"""Unit tests for EnclaveJobInfo, the per-party approval gate.

The gate lives here rather than in SyftEnclaveClient.approve_job, so these
build a job on a tmp_path SyftBox folder instead of a four-party enclave flow.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from syft_enclaves.enclave_job_info import (
    EnclaveJobInfo,
    PartyApprovalStatus,
    enclave_approval_file_name,
)
from syft_job.client import JobClient
from syft_job.config import SyftJobConfig
from syft_job.job import JobInfo
from syft_job.job_storage import JobRef
from syft_job.models import JobState, JobStatus, JobSubmissionMetadata

DO_EMAIL = "do@test.org"
DS_EMAIL = "ds@test.org"


def _make_enclave_job(tmp_path: Path, job_name: str = "test_job") -> EnclaveJobInfo:
    """An enclave job on the DO's datasite, with no approval file written yet."""
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    client = JobClient(
        config=SyftJobConfig(syftbox_folder=syftbox, current_user_email=DO_EMAIL)
    )
    ref = JobRef(
        datasite_email=DO_EMAIL,
        ds_email=DS_EMAIL,
        job_name=job_name,
        protocol_version="1",
    )
    job = JobInfo(
        job_metadata=JobSubmissionMetadata(
            name=job_name,
            type="python",
            submitted_by=DS_EMAIL,
            datasite_email=DO_EMAIL,
            submitted_at=datetime.now(timezone.utc),
        ),
        state=JobState(status=JobStatus.PENDING),
        client=client,
        current_user_email=DO_EMAIL,
        ref=ref,
    )
    return EnclaveJobInfo.from_job_info(job)


def test_approve_refuses_when_approval_file_missing(tmp_path: Path):
    """No approval file means the enclave has not distributed the job yet.

    The message used to say the caller may not be a designated party, which is
    the wrong cause for the common case and offers nothing to do about it.
    """
    job = _make_enclave_job(tmp_path)

    with pytest.raises(PermissionError) as exc:
        job.approve()

    message = str(exc.value)
    assert DO_EMAIL in message
    assert "test_job" in message
    assert "client.sync()" in message


def test_approve_refuses_when_already_approved(tmp_path: Path):
    """A second approval must not overwrite the first one's timestamp."""
    job = _make_enclave_job(tmp_path)
    approval_file = job.job_review_path / enclave_approval_file_name(DO_EMAIL)
    PartyApprovalStatus(party=DO_EMAIL).save_json(approval_file)

    job.approve()
    assert PartyApprovalStatus.load_json(approval_file).status == JobStatus.APPROVED

    with pytest.raises(ValueError, match="Already in status: approved"):
        job.approve()
