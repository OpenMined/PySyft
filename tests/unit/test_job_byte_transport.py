"""Keep the receipt hash contract through filesystem-backed mock Drive sync."""

from pathlib import Path

import pytest
from syft_job.client import JobClient
from syft_job.config import SyftJobConfig
from syft_job.submission import SUBMISSION_HASH_HEADER, code_hash

from syft.sync.syftbox_manager import SyftboxManager

TransportPair = tuple[JobClient, JobClient, SyftboxManager, SyftboxManager]


@pytest.fixture
def transport(tmp_path: Path) -> TransportPair:
    ds_sync, do_sync = SyftboxManager.pair_with_mock_drive_service_connection(
        email1="do@test.org",
        email2="ds@test.org",
        base_path1=str(tmp_path / "owner"),
        base_path2=str(tmp_path / "submitter"),
        use_in_memory_cache=False,
        persist_owner_state=False,
        sync_automatically=False,
    )
    do = JobClient(
        SyftJobConfig(
            syftbox_folder=do_sync.syftbox_folder, current_user_email=do_sync.email
        )
    )
    ds = JobClient(
        SyftJobConfig(
            syftbox_folder=ds_sync.syftbox_folder, current_user_email=ds_sync.email
        )
    )
    do.setup_ds_job_folder_as_do(ds_sync.email)
    return do, ds, do_sync, ds_sync


@pytest.mark.parametrize(
    "payload", [b"caf\xc3\xa9\r\n", b"\xff\x00\r\n\x80"], ids=["utf8-crlf", "binary"]
)
def test_job_transport_preserves_raw_declared_bytes(
    transport: TransportPair, tmp_path: Path, payload: bytes
) -> None:
    do, ds, do_sync, ds_sync = transport
    source = tmp_path / "source"
    source.mkdir()
    code = b"# caf\xc3\xa9\r\nprint('approved')\r\n"
    (source / "main.py").write_bytes(code)
    (source / "payload.bin").write_bytes(payload)
    inbox = ds.submit_python_job(do_sync.email, str(source), "transport.job")
    [ref] = list(ds.manager.iter_submission_refs(do_sync.email))
    declared = ds.manager.read_submission(ref).headers[SUBMISSION_HASH_HEADER]

    ds_sync.push_job_files(inbox)
    do_sync.sync()
    [job] = do.jobs

    assert job.status == "pending"
    assert (job.job_submission_path / "code/main.py").read_bytes() == code
    assert (job.job_submission_path / "code/payload.bin").read_bytes() == payload
    assert code_hash(job.job_submission_path) == declared
    assert (inbox / "code/main.py").read_bytes() == code


def test_job_transport_refuses_newline_change_after_declaration(
    transport: TransportPair, tmp_path: Path
) -> None:
    do, ds, do_sync, ds_sync = transport
    source = tmp_path / "main.py"
    source.write_bytes(b"print('approved')\n")
    inbox = ds.submit_python_job(do_sync.email, str(source), "changed.job")
    changed = inbox / "code/main.py"
    changed.write_bytes(changed.read_bytes().replace(b"\n", b"\r\n"))

    ds_sync.push_job_files(inbox)
    do_sync.sync()

    [job] = do.jobs
    assert job.status == "received"
