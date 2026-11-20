"""Receipt, approval and execution must agree on the submitted bytes."""

from pathlib import Path

import pytest
from syft_job.client import JobClient
from syft_job.config import SyftJobConfig
from syft_job.job_runner import SyftJobRunner
from syft_job.policies import JobPolicies, ResolvedApi
from syft_job.submission import request_hash, submission_hash

import syft

DO_EMAIL = "do@test.org"
DS_EMAIL = "ds@test.org"
ClientPair = tuple[JobClient, JobClient]


@pytest.fixture
def clients(tmp_path: Path) -> ClientPair:
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    do = JobClient(SyftJobConfig(syftbox_folder=syftbox, current_user_email=DO_EMAIL))
    ds = JobClient(SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS_EMAIL))
    ds.peer_install_sources[DO_EMAIL] = str(Path(syft.__file__).resolve().parents[1])
    do.setup_ds_job_folder_as_do(DS_EMAIL)
    return do, ds


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"], ids=["lf", "crlf"])
def test_raw_bytes_bind_receipt_approval_and_execution(
    clients: ClientPair, tmp_path: Path, newline: bytes
) -> None:
    do, ds = clients
    source = tmp_path / "source"
    source.mkdir()
    code = (
        (
            "from pathlib import Path\n"
            "assert Path(__file__).with_name('payload.bin').read_bytes() == "
            "b'\\xff\\x00\\r\\n\\x80'\n"
            "print('données approuvées')\n"
        )
        .encode("utf-8")
        .replace(b"\n", newline)
    )
    (source / "main.py").write_bytes(code)
    (source / "payload.bin").write_bytes(b"\xff\x00\r\n\x80")
    inbox = ds.submit_python_job(DO_EMAIL, str(source), "bytes.job")

    [job] = do.jobs
    assert job.status == "pending"
    record = do.manager.read_owner_record(job._ref)
    assert record is not None
    received = submission_hash(inbox)
    assert record.received_hash == received
    job.approve()
    record = do.manager.read_owner_record(job._ref)
    assert record is not None
    assert record.approved_hash == record.received_hash == received

    SyftJobRunner(do.config).process_approved_jobs(stream_output=False, timeout=60)

    [job] = do.jobs
    assert job.status == "done"
    assert (job.job_staging_path / "stdout.txt").read_text().strip() == (
        "données approuvées"
    )
    assert (inbox / "code/main.py").read_bytes() == code
    assert (inbox / "code/payload.bin").read_bytes() == b"\xff\x00\r\n\x80"


@pytest.mark.parametrize("change", ["newline", "binary"])
@pytest.mark.parametrize(
    "when", ["before_receipt", "before_approval", "after_approval"]
)
def test_changed_bytes_never_execute(
    clients: ClientPair, tmp_path: Path, when: str, change: str
) -> None:
    do, ds = clients
    source = tmp_path / "source"
    source.mkdir()
    marker = tmp_path / "executed.txt"
    (source / "main.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
    )
    (source / "payload.bin").write_bytes(b"\xff\x00\r\n\x80")
    inbox = ds.submit_python_job(DO_EMAIL, str(source), "changed.job")
    if when != "before_receipt":
        [job] = do.jobs
        assert job.status == "pending"
        if when == "after_approval":
            job.approve()

    if change == "newline":
        target = inbox / "code/main.py"
        target.write_bytes(target.read_bytes().replace(b"\n", b"\r\n"))
    else:
        (inbox / "code/payload.bin").write_bytes(b"\xff\x00\n\x80")

    if when == "before_receipt":
        [received_job] = do.jobs
        assert received_job.status == "received"
    elif when == "before_approval":
        with pytest.raises(ValueError, match="changed after it was received"):
            job.approve()
    else:
        SyftJobRunner(do.config).process_approved_jobs(stream_output=False, timeout=10)
        [failed_job] = do.jobs
        assert failed_job.status == "failed"
    assert not marker.exists()


@pytest.mark.parametrize(
    "payload",
    [b'{"word":"caf\xc3\xa9"}\r\n', b"\xff\x00\r\n\x80"],
    ids=["utf8-crlf", "binary"],
)
def test_request_bytes_survive_conversion_to_approved_job(
    clients: ClientPair, tmp_path: Path, payload: bytes
) -> None:
    do, ds = clients
    files = tmp_path / "api"
    files.mkdir()
    (files / "run.sh").write_text("echo approved\n")
    policies = JobPolicies(api_name="bytes", state_dir=tmp_path / "policy")

    class ByteApi:
        def __call__(self, api_name: str) -> ResolvedApi:
            assert api_name == "bytes"
            return ResolvedApi(api_name, files, policies)

        def check_request(
            self, api_name: str, request_dir: Path, submitter: str
        ) -> None:
            self(api_name)

    do.api_resolver = ByteApi()
    request_dir = ds.submit_api_request(
        DO_EMAIL, "bytes.job", "bytes", {"code/payload.bin": payload}
    )
    [ref] = list(ds.manager.iter_request_refs(DO_EMAIL))
    assert ds.manager.read_request(ref).files_hash == request_hash(request_dir)

    jobs = do.jobs
    assert len(jobs) == 1
    [job] = jobs
    assert job.status == "approved"
    assert (job.job_submission_path / "code/payload.bin").read_bytes() == payload
    record = do.manager.read_owner_record(job._ref)
    assert record is not None
    assert record.approved_hash == record.received_hash
    assert record.approved_hash == submission_hash(job.job_submission_path)
