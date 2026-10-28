"""An api call (a request) in requests/, and how the DO turns it into a job."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from syft_migration import ProtocolSchema
from syft_perms import SyftPermContext

from syft_job.client import JobClient
from syft_job.config import SyftJobConfig
from syft_job.policies import JobPolicies, PolicyResolutionError, ResolvedApi
from syft_job.models import RequestMetadata
from syft_job.submission import (
    request_hash,
)
from syft_job.config import protocol_dir_name
from syft_job.migrations.registry import JOB_PROTOCOL_VERSION

# The path segment of the current job protocol, e.g. "v2".
PROTOCOL_DIR = protocol_dir_name(JOB_PROTOCOL_VERSION)

DO_EMAIL = "do@test.org"
DS_EMAIL = "ds@test.org"
API_NAME = "word_count"
PARAMS = {"code/params.json": b'{"word": "privacy"}'}
REQUESTS_REL = f"app_data/job/requests/{DS_EMAIL}/{PROTOCOL_DIR}"
INBOX_REL = f"app_data/job/inbox/{DS_EMAIL}/{PROTOCOL_DIR}"


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
    assert _ds_writes(do, f"{REQUESTS_REL}/call.job/request.yaml")

    ds.submit_api_request(DO_EMAIL, "call.job", API_NAME, PARAMS)

    assert len(ds.jobs) == 0
    [sent] = ds.requests
    assert (sent.name, sent.status, sent.api_name) == ("call.job", "received", API_NAME)
    assert sent.job is None


def test_request_becomes_approved_locked_job(env):
    do, ds, _ = env
    ds.submit_api_request(DO_EMAIL, "call.job", API_NAME, PARAMS)

    [job] = do.jobs

    assert (job.status, job.approval_method, job.api_name) == (
        "approved",
        "api",
        API_NAME,
    )
    assert ds.requests == []
    assert (job.job_submission_path / "run.sh").read_text() == "echo hi\n"
    assert (job.job_submission_path / "code" / "params.json").exists()
    assert not (_datasite(do) / REQUESTS_REL / "call.job").exists()
    assert not _ds_writes(do, f"{INBOX_REL}/call.job/run.sh")


def test_request_with_symlink_is_rejected_and_kept(env):
    """A symlink in a request could point at any file on the DO's machine."""
    do, ds, _ = env
    request_dir = ds.submit_api_request(DO_EMAIL, "call.job", API_NAME, PARAMS)
    (request_dir / "code" / "link.txt").symlink_to("/etc/hosts")
    [ref] = list(ds.manager.iter_request_refs(DO_EMAIL))
    request = ds.manager.read_request(ref)
    request.files_hash = request_hash(request_dir)
    ds.manager.write_request(ref, request)

    [request] = do.requests

    assert len(do.jobs) == 0 and request.status == "rejected"
    assert "symlink" in request.review_reason
    assert not (_datasite(do) / INBOX_REL / "call.job").exists()
    assert not _ds_writes(do, f"{REQUESTS_REL}/call.job/request.yaml")


def test_request_may_not_send_config(env):
    _, ds, _ = env
    for reserved in ("config.yaml", "request.yaml"):
        with pytest.raises(ValueError, match=reserved):
            ds.submit_api_request(DO_EMAIL, "call.job", API_NAME, {reserved: b""})


def test_request_refused_by_api_is_rejected(env):
    do, ds, api = env
    api.refused = "params don't fit the api: 'word' must be str"
    ds.submit_api_request(DO_EMAIL, "call.job", API_NAME, PARAMS)

    [request] = do.requests

    assert request.status == "rejected"
    assert request.review_reason == "params don't fit the api: 'word' must be str"


def test_client_refuses_request_named_like_job(env):
    _, ds, _ = env
    ds.submit_bash_job(DO_EMAIL, "echo hi\n", "same.job")

    with pytest.raises(FileExistsError):
        ds.submit_api_request(DO_EMAIL, "same.job", API_NAME, PARAMS)


def _write_request_by_hand(ds: JobClient, job_name: str) -> None:
    """A request as submit_api_request writes it, without its name check."""
    ref = ds.manager.new_request_ref(DO_EMAIL, job_name)
    request_dir = ds.manager.request_dir(ref)
    (request_dir / "code").mkdir(parents=True)
    (request_dir / "code" / "params.json").write_bytes(PARAMS["code/params.json"])
    request = RequestMetadata(
        api=API_NAME,
        submitted_at=datetime.now(timezone.utc),
        files=sorted(PARAMS),
        files_hash=request_hash(request_dir),
    )
    ds.manager.write_request(ref, request)


@pytest.mark.parametrize("job_received_first", [True, False])
def test_request_named_like_job_leaves_job_alone(env, job_received_first):
    do, ds, _ = env
    ds.submit_bash_job(DO_EMAIL, "echo hi\n", "same.job")
    if job_received_first:
        do.scan_inbox()
    _write_request_by_hand(ds, "same.job")

    [whole_job] = do.jobs

    assert whole_job.status == "pending" and whole_job.api_name is None
    assert not _ds_writes(do, f"{REQUESTS_REL}/same.job/code/params.json")


def test_request_waits_for_client_that_loads_apis(env):
    do, ds, _ = env
    do.api_resolver = None
    ds.submit_api_request(DO_EMAIL, "call.job", API_NAME, PARAMS)

    [request] = do.requests

    assert len(do.jobs) == 0 and request.status == "received"


def test_client_refuses_request_to_do_without_api_calls(tmp_path: Path):
    """A DO that speaks job protocol 1 has no requests/ to receive the call."""
    syftbox = tmp_path / "SyftBox"
    config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS_EMAIL)
    old_do = ProtocolSchema(protocol_name="syft-job", version="1")
    ds = JobClient(config, peer_schema={DO_EMAIL: old_do}.get)

    with pytest.raises(ValueError, match="job protocol 1"):
        ds.submit_api_request(DO_EMAIL, "call.job", API_NAME, PARAMS)
