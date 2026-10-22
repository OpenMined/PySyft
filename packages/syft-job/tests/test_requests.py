"""An api call (a request) in requests/, and how the DO turns it into a job."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from syft_perms import SyftPermContext

from syft_job.client import JobClient
from syft_job.config import SyftJobConfig
from syft_job.models import JobSubmissionMetadata
from syft_job.policies import JobPolicies, PolicyResolutionError, ResolvedApi
from syft_job.submission import (
    API_HEADER,
    SUBMISSION_HASH_HEADER,
    request_hash,
)

DO_EMAIL = "do@test.org"
DS_EMAIL = "ds@test.org"
API_NAME = "word_count"
PARAMS = {"code/params.json": b'{"word": "privacy"}'}
REQUESTS_REL = f"app_data/job/requests/{DS_EMAIL}/v1"
INBOX_REL = f"app_data/job/inbox/{DS_EMAIL}/v1"


class OneApi:
    """An api whose code is one run.sh; refused set makes check_request fail."""

    def __init__(self, tmp_path: Path):
        self.files_dir = tmp_path / "api_files"
        self.files_dir.mkdir()
        (self.files_dir / "run.sh").write_text("echo hi\n")
        self.state_dir = tmp_path / "state"
        self.refused: str | None = None

    def __call__(self, api_name: str) -> ResolvedApi:
        if api_name != API_NAME:
            raise PolicyResolutionError(f"no api named {api_name!r}")
        policies = JobPolicies(api_name=api_name, state_dir=self.state_dir)
        return ResolvedApi(api_name, self.files_dir, policies)

    def check_request(self, api_name: str, request_dir: Path, submitter: str) -> None:
        if self.refused:
            raise PolicyResolutionError(self.refused)
        self(api_name)


@pytest.fixture
def env(tmp_path: Path):
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    do = JobClient(SyftJobConfig(syftbox_folder=syftbox, current_user_email=DO_EMAIL))
    ds = JobClient(SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS_EMAIL))
    do.setup_ds_job_folder_as_do(DS_EMAIL)
    api = OneApi(tmp_path)
    do.api_resolver = api
    return do, ds, api


def _ds_writes(do: JobClient, rel_path: str) -> bool:
    ctx = SyftPermContext(datasite=do.config.syftbox_folder / DO_EMAIL)
    return ctx.open(rel_path).has_write_access(DS_EMAIL)


def _datasite(do: JobClient) -> Path:
    return do.config.syftbox_folder / DO_EMAIL


def test_submitter_finds_request_before_receipt(env):
    do, ds, _ = env
    assert _ds_writes(do, f"{REQUESTS_REL}/call.job/config.yaml")

    ds.submit_api_request(DO_EMAIL, "call.job", API_NAME, PARAMS)

    [sent] = ds.jobs
    assert (sent.name, sent.status) == ("call.job", "received")
    assert sent.is_request and sent.api_name == API_NAME


def test_request_becomes_approved_locked_job(env):
    do, ds, _ = env
    ds.submit_api_request(DO_EMAIL, "call.job", API_NAME, PARAMS)

    [job] = do.jobs

    assert (job.status, job.approval_method, job.api_name) == (
        "approved",
        "api",
        API_NAME,
    )
    assert not job.is_request
    assert (job.job_submission_path / "run.sh").read_text() == "echo hi\n"
    assert (job.job_submission_path / "code" / "params.json").exists()
    assert not (_datasite(do) / REQUESTS_REL / "call.job").exists()
    assert not _ds_writes(do, f"{INBOX_REL}/call.job/run.sh")


def test_request_with_symlink_is_rejected_and_kept(env):
    """A symlink in a request could point at any file on the DO's machine."""
    do, ds, _ = env
    request_dir = ds.submit_api_request(DO_EMAIL, "call.job", API_NAME, PARAMS)
    (request_dir / "code" / "link.txt").symlink_to("/etc/hosts")
    [ref] = list(ds.manager.iter_submission_refs(DO_EMAIL))
    config = ds.manager.read_submission(ref)
    config.headers[SUBMISSION_HASH_HEADER] = request_hash(request_dir, as_sent=True)
    ds.manager.write_submission(ref, config)

    [job] = do.jobs

    assert job.is_request and job.status == "rejected"
    assert "symlink" in job.review_reason
    assert not (_datasite(do) / INBOX_REL / "call.job").exists()
    assert not _ds_writes(do, f"{REQUESTS_REL}/call.job/config.yaml")


def test_request_may_not_send_config(env):
    _, ds, _ = env
    with pytest.raises(ValueError, match="config.yaml"):
        ds.submit_api_request(DO_EMAIL, "call.job", API_NAME, {"config.yaml": b""})


def test_request_refused_by_api_is_rejected(env):
    do, ds, api = env
    api.refused = "params don't fit the api: 'word' must be str"
    ds.submit_api_request(DO_EMAIL, "call.job", API_NAME, PARAMS)

    [job] = do.jobs

    assert job.status == "rejected"
    assert job.review_reason == "params don't fit the api: 'word' must be str"


def test_client_refuses_request_named_like_job(env):
    _, ds, _ = env
    ds.submit_bash_job(DO_EMAIL, "echo hi\n", "same.job")

    with pytest.raises(FileExistsError):
        ds.submit_api_request(DO_EMAIL, "same.job", API_NAME, PARAMS)


def _write_request_by_hand(ds: JobClient, job_name: str) -> None:
    """A request as submit_api_request writes it, without its name check."""
    ref = ds.manager.new_submission_ref(DO_EMAIL, job_name, is_request=True)
    request_dir = ds.manager.submission_dir(ref)
    (request_dir / "code").mkdir(parents=True)
    (request_dir / "code" / "params.json").write_bytes(PARAMS["code/params.json"])
    headers = {
        API_HEADER: API_NAME,
        SUBMISSION_HASH_HEADER: request_hash(request_dir, as_sent=True),
    }
    config = JobSubmissionMetadata(
        name=job_name,
        submitted_by=DS_EMAIL,
        datasite_email=DO_EMAIL,
        submitted_at=datetime.now(timezone.utc),
        headers=headers,
    )
    ds.manager.write_submission(ref, config)


@pytest.mark.parametrize("job_received_first", [True, False])
def test_request_named_like_job_leaves_job_alone(env, job_received_first):
    do, ds, _ = env
    ds.submit_bash_job(DO_EMAIL, "echo hi\n", "same.job")
    if job_received_first:
        do.scan_inbox()
    _write_request_by_hand(ds, "same.job")

    jobs = do.jobs

    whole_job = next(j for j in jobs if not j.is_request)
    assert whole_job.status == "pending" and whole_job.api_name is None
    assert not _ds_writes(do, f"{REQUESTS_REL}/same.job/code/params.json")


def test_request_waits_for_client_that_loads_apis(env):
    do, ds, _ = env
    do.api_resolver = None
    ds.submit_api_request(DO_EMAIL, "call.job", API_NAME, PARAMS)

    [job] = do.jobs

    assert job.is_request and job.status == "received"
