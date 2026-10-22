"""Receiving an api call (a request) from requests/, before anything runs."""

from pathlib import Path

import pytest
from syft_perms import SyftPermContext

from syft_job.client import JobClient
from syft_job.config import SyftJobConfig
from syft_job.submission import SUBMISSION_HASH_HEADER, code_hash

DO_EMAIL = "do@test.org"
DS_EMAIL = "ds@test.org"
API_NAME = "word_count"
PARAMS = {"params.json": b'{"word": "privacy"}'}
REQUESTS_REL = f"app_data/job/requests/{DS_EMAIL}"


@pytest.fixture
def clients(tmp_path: Path) -> tuple[JobClient, JobClient]:
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    do = JobClient(SyftJobConfig(syftbox_folder=syftbox, current_user_email=DO_EMAIL))
    ds = JobClient(SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS_EMAIL))
    do.setup_ds_job_folder_as_do(DS_EMAIL)
    return do, ds


def _ds_writes(do: JobClient, rel_path: str) -> bool:
    ctx = SyftPermContext(datasite=do.config.syftbox_folder / DO_EMAIL)
    return ctx.open(rel_path).has_write_access(DS_EMAIL)


def test_request_is_listed_and_locked_once_received(clients):
    do, ds = clients
    assert _ds_writes(do, f"{REQUESTS_REL}/v1/call.job/config.yaml")
    ds.submit_api_request(DO_EMAIL, "call.job", API_NAME, PARAMS)

    # The submitter finds the call by its name before the DO received it.
    [sent] = ds.jobs
    assert (sent.name, sent.status) == ("call.job", "received")
    assert sent.is_request and sent.api_name == API_NAME

    [received] = do.jobs
    assert received.status == "pending" and received.is_request
    assert not _ds_writes(do, f"{REQUESTS_REL}/v1/call.job/code/params.json")


def test_request_with_code_is_rejected_at_receipt(clients):
    """A submitter that adds run.sh to a request, and declares its hash."""
    do, ds = clients
    job_dir = ds.submit_api_request(DO_EMAIL, "call.job", API_NAME, PARAMS)
    (job_dir / "run.sh").write_text("echo injected\n")
    [ref] = list(ds.manager.iter_submission_refs(DO_EMAIL))
    config = ds.manager.read_submission(ref)
    config.headers[SUBMISSION_HASH_HEADER] = code_hash(job_dir, as_sent=True)
    ds.manager.write_submission(ref, config)

    [job] = do.jobs

    assert job.status == "rejected"
    assert "Expected" in job.review_reason


def test_request_named_like_job_is_not_received(clients):
    do, ds = clients
    ds.submit_bash_job(DO_EMAIL, "echo hi\n", "same.job")
    ds.submit_api_request(DO_EMAIL, "same.job", API_NAME, PARAMS)

    jobs = do.jobs

    whole_job = next(j for j in jobs if not j.is_request)
    request = next(j for j in jobs if j.is_request)
    assert whole_job.status == "pending"
    assert do.manager.read_owner_record(request._ref).is_request is False


def test_request_needs_api_resolver_to_approve(clients):
    do, ds = clients
    ds.submit_api_request(DO_EMAIL, "call.job", API_NAME, PARAMS)
    [job] = do.jobs

    with pytest.raises(ValueError, match="can't load apis"):
        job.approve()
    assert do.jobs[0].status == "pending"
