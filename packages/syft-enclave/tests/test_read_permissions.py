"""Per-file read permissions: the enclave shares a file only under full agreement."""

import inspect
import json
import os
import random
import tempfile
import warnings
from pathlib import Path

import pytest
import yaml

os.environ["PRE_SYNC"] = "false"

from syft_job.job import JobInfo  # noqa: E402
from syft_job.models import JobState, JobStatus  # noqa: E402
from syft_job.read_permissions import (  # noqa: E402
    LOGS_WARNING,
    OUTPUTS,
    REQUESTED_READ_PERMISSIONS_HEADER,
)
from syft_perms import SyftPermContext  # noqa: E402

from syft_enclaves import SyftEnclaveClient  # noqa: E402
from syft_enclaves.enclave_job_info import (  # noqa: E402
    EnclaveJobInfo,
    PartyApprovalStatus,
    enclave_approval_file_name,
    read_permissions_approved_by_all_parties,
)
from syft_enclaves.log_requests import (  # noqa: E402
    LogRequest,
    RequestedPermissionsUpdate,
    log_request_file_name,
    logs_requested_by_all_parties,
)

LOGS = ("stdout.txt", "stderr.txt")
RETURN_CODE = "returncode.txt"
DS = "ds@x.com"


def readable(files, *emails):
    """Read permissions that let ``emails`` read ``files`` (one name or several)."""
    names = [files] if isinstance(files, str) else files
    return {name: sorted(emails) for name in names}


def write_approval(review_dir, party, status, read_permissions):
    approval = PartyApprovalStatus(
        party=party, status=status, approved_read_permissions=read_permissions or {}
    )
    approval.save_json(review_dir / enclave_approval_file_name(party))


# -- the resolution rule --------------------------------------------------------


def test_file_needs_every_party(tmp_path):
    write_approval(tmp_path, "do1@x.com", JobStatus.APPROVED, readable(LOGS, DS))
    write_approval(tmp_path, "do2@x.com", JobStatus.APPROVED, readable(LOGS, DS))
    approved = read_permissions_approved_by_all_parties(tmp_path, readable(LOGS, DS))
    assert approved == readable(LOGS, DS)


def test_one_party_withholding_blocks_file(tmp_path):
    write_approval(tmp_path, "do1@x.com", JobStatus.APPROVED, readable(LOGS, DS))
    write_approval(tmp_path, "do2@x.com", JobStatus.APPROVED, readable(RETURN_CODE, DS))
    requested = readable([*LOGS, RETURN_CODE], DS)
    assert read_permissions_approved_by_all_parties(tmp_path, requested) == {}


def test_reader_needs_every_party(tmp_path):
    """Each email is agreed on its own: one party leaving it out keeps it out."""
    both = readable(RETURN_CODE, DS, "do1@x.com")
    write_approval(tmp_path, "do1@x.com", JobStatus.APPROVED, both)
    write_approval(tmp_path, "do2@x.com", JobStatus.APPROVED, readable(RETURN_CODE, DS))
    approved = read_permissions_approved_by_all_parties(tmp_path, both)
    assert approved == readable(RETURN_CODE, DS)


def test_unapproved_party_blocks_everything(tmp_path):
    write_approval(tmp_path, "do1@x.com", JobStatus.APPROVED, readable(LOGS, DS))
    write_approval(tmp_path, "do2@x.com", JobStatus.PENDING, readable(LOGS, DS))
    assert read_permissions_approved_by_all_parties(tmp_path, readable(LOGS, DS)) == {}


def test_approval_file_without_read_permissions_grants_nothing(tmp_path):
    """An older client writes no approved_read_permissions key, so it approves nothing."""
    path = tmp_path / enclave_approval_file_name("do1@x.com")
    path.write_text(json.dumps({"party": "do1@x.com", "status": "approved"}))
    assert PartyApprovalStatus.load_json(path).approved_read_permissions == {}
    assert read_permissions_approved_by_all_parties(tmp_path, readable(LOGS, DS)) == {}


def test_request_narrows_result(tmp_path):
    approved = readable([*LOGS, RETURN_CODE], DS)
    write_approval(tmp_path, "do1@x.com", JobStatus.APPROVED, approved)
    requested = readable(RETURN_CODE, DS)
    assert read_permissions_approved_by_all_parties(tmp_path, requested) == requested


def test_unrequested_reader_never_survives(tmp_path):
    """Every party approving an email the submitter did not name gives it nothing."""
    wider = readable(RETURN_CODE, DS, "other@x.com")
    write_approval(tmp_path, "do1@x.com", JobStatus.APPROVED, wider)
    write_approval(tmp_path, "do2@x.com", JobStatus.APPROVED, wider)
    requested = readable(RETURN_CODE, DS)
    assert read_permissions_approved_by_all_parties(tmp_path, requested) == requested


def test_no_approval_file_grants_nothing(tmp_path):
    assert read_permissions_approved_by_all_parties(tmp_path, readable(LOGS, DS)) == {}


def test_named_party_without_file_grants_nothing(tmp_path):
    """A missing file must not leave the other approvals standing alone."""
    logs = readable(LOGS, DS)
    write_approval(tmp_path, "do1@x.com", JobStatus.APPROVED, logs)
    assert read_permissions_approved_by_all_parties(tmp_path, logs) == logs
    parties = ["do1@x.com", "do2@x.com"]
    assert read_permissions_approved_by_all_parties(tmp_path, logs, parties) == {}


def test_no_named_party_grants_nothing(tmp_path):
    logs = readable(LOGS, DS)
    write_approval(tmp_path, "do1@x.com", JobStatus.APPROVED, logs)
    assert read_permissions_approved_by_all_parties(tmp_path, logs, parties=[]) == {}


def test_older_client_item_approves_nothing(tmp_path):
    """An item of an older client, e.g. {"logs": true}, approves nothing."""
    path = tmp_path / enclave_approval_file_name("do1@x.com")
    approved = {"logs": True}
    path.write_text(
        json.dumps(
            {
                "party": "do1@x.com",
                "status": "approved",
                "approved_read_permissions": approved,
            }
        )
    )
    approval = PartyApprovalStatus.load_json(path)
    assert approval.status == JobStatus.APPROVED
    assert approval.approved_read_permissions == {}
    assert read_permissions_approved_by_all_parties(tmp_path, readable(LOGS, DS)) == {}


def test_malformed_readers_grant_nothing(tmp_path):
    """A string of readers is not iterated into characters, and no list is empty."""
    path = tmp_path / enclave_approval_file_name("do1@x.com")
    approved = {"stdout.txt": [], "stderr.txt": DS, RETURN_CODE: [DS]}
    path.write_text(
        json.dumps(
            {
                "party": "do1@x.com",
                "status": "approved",
                "approved_read_permissions": approved,
            }
        )
    )
    loaded = PartyApprovalStatus.load_json(path).approved_read_permissions
    assert loaded == readable(RETURN_CODE, DS)


# -- end to end -----------------------------------------------------------------


def build_quad():
    enclave, do1, do2, ds = SyftEnclaveClient.quad_with_mock_drive_service_connection(
        use_in_memory_cache=False
    )
    for do, name in ((do1, "dataset1"), (do2, "dataset2")):
        d = Path(tempfile.mkdtemp()) / f"d{random.randint(1, 10**6)}"
        d.mkdir(parents=True)
        (d / "m.txt").write_text("m")
        (d / "p.txt").write_text("p")
        do.create_dataset(
            name=name,
            mock_path=d / "m.txt",
            private_path=d / "p.txt",
            summary=name,
            users=[ds.email, enclave.email],
            upload_private=True,
            sync=False,
        )
        do.share_private_dataset(name, enclave.email)
        do.sync()
    ds.sync()
    return enclave, do1, do2, ds


def submit(ds, enclave, do1, do2, code, requested=None, datasets=None, **share_flags):
    path = Path(tempfile.mkdtemp()) / "main.py"
    path.write_text(code)
    ds.submit_python_job(
        enclave.email,
        str(path),
        "j",
        datasets=datasets or {do1.email: ["dataset1"], do2.email: ["dataset2"]},
        share_files=requested,
        **share_flags,
    )


def distribute(enclave, *dos):
    """The enclave receives and distributes the job, and each DO syncs it."""
    enclave.sync()
    enclave.receive_jobs()
    for do in dos:
        do.sync()


OK_CODE = "import os, json\nos.makedirs('outputs', exist_ok=True)\nopen('outputs/r.json','w').write('{}')\n"
RESULT = "outputs/r.json"
CRASH_CODE = "x = 1\ny = 2\nraise ValueError('account 88213 holds 4120550')\n"


# The DO calls the job, or the client method that wraps it.
ENTRY_POINTS = pytest.mark.parametrize("via_job", [False, True], ids=["client", "job"])


def approve(do, approved, via_job=False):
    """``{}`` approves nothing; None approves the whole request."""
    if via_job:
        do.jobs["j"].approve(share_files=approved)
    else:
        do.approve_job(do.jobs["j"], approved)


def update(do, approved, via_job=False):
    if via_job:
        return do.jobs["j"].update_read_permissions(approved)
    return do.update_read_permissions(do.jobs["j"], approved)


def run_to_completion(enclave, do1, do2, ds, approved, via_job=False):
    distribute(enclave, do1, do2)
    approve(do1, approved, via_job)
    approve(do2, approved, via_job)
    run_and_share(enclave, ds)


def run_and_share(enclave, *readers):
    """The enclave runs the job and shares what was approved; each reader syncs."""
    enclave.sync()
    enclave.run_jobs()
    enclave.distribute_results()
    for reader in readers:
        reader.sync()


def has_file(client, name):
    return (Path(client.jobs["j"].job_review_path) / name).exists()


def readers_of(enclave, name, parties):
    """The parties that may read ``name`` in the enclave's review folder."""
    datasite = enclave._rds.syftbox_folder / enclave.email
    review = Path(enclave._rds.job_client.jobs["j"].job_review_path)
    path = str((review / name).relative_to(datasite))
    ctx = SyftPermContext(datasite=datasite)
    return [p for p in parties if ctx.can_read(path, p.email)]


def test_submitter_gets_no_logs_without_approval():
    enclave, do1, do2, ds = build_quad()
    submit(ds, enclave, do1, do2, OK_CODE, readable(LOGS, ds.email))
    run_to_completion(enclave, do1, do2, ds, {})

    assert not has_file(ds, "stdout.txt")
    assert not has_file(ds, "stderr.txt")


@ENTRY_POINTS
def test_full_approval_sends_logs_to_submitter(via_job):
    enclave, do1, do2, ds = build_quad()
    logs = readable(LOGS, ds.email)
    submit(ds, enclave, do1, do2, OK_CODE, logs)
    run_to_completion(enclave, do1, do2, ds, logs, via_job)

    assert enclave.read_permissions_to_share(enclave.jobs["j"]) == logs
    assert has_file(ds, "stdout.txt")
    # The enclave records what it shared in the state, which the submitter reads.
    assert ds.jobs["j"].approved_read_permissions == logs


def test_submitter_gets_no_return_code_without_approval():
    enclave, do1, do2, ds = build_quad()
    submit(ds, enclave, do1, do2, CRASH_CODE, readable(RETURN_CODE, ds.email))
    run_to_completion(enclave, do1, do2, ds, readable(LOGS, ds.email))

    review = Path(ds.jobs["j"].job_review_path)
    assert not (review / RETURN_CODE).exists()
    state = JobState.load(review / "state.yaml")
    assert state.status == JobStatus.FAILED
    assert state.return_code is None


def test_return_code_approval_sends_only_exit_code():
    enclave, do1, do2, ds = build_quad()
    return_code = readable(RETURN_CODE, ds.email)
    submit(ds, enclave, do1, do2, CRASH_CODE, return_code)
    run_to_completion(enclave, do1, do2, ds, return_code)

    review = Path(ds.jobs["j"].job_review_path)
    assert (review / RETURN_CODE).read_text().strip() == "1"
    assert not (review / "stderr.txt").exists()


def test_return_code_reaches_named_data_owner():
    """A data owner that the request names, and every party approved, receives the file."""
    enclave, do1, do2, ds = build_quad()
    return_code = readable(RETURN_CODE, ds.email, do1.email)
    submit(ds, enclave, do1, do2, CRASH_CODE, return_code)
    run_to_completion(enclave, do1, do2, ds, return_code)
    do1.sync()

    assert has_file(do1, RETURN_CODE)


def test_unnamed_data_owner_gets_no_file():
    """Every party approving the submitter's read gives no data owner a copy."""
    enclave, do1, do2, ds = build_quad()
    return_code = readable(RETURN_CODE, ds.email)
    submit(ds, enclave, do1, do2, CRASH_CODE, return_code)
    run_to_completion(enclave, do1, do2, ds, return_code)
    do1.sync()
    do2.sync()

    assert has_file(ds, RETURN_CODE)
    assert not has_file(do1, RETURN_CODE)
    assert not has_file(do2, RETURN_CODE)


def test_approved_file_grants_read_to_named_parties():
    enclave, do1, do2, ds = build_quad()
    return_code = readable(RETURN_CODE, ds.email, do1.email)
    submit(ds, enclave, do1, do2, CRASH_CODE, return_code)
    run_to_completion(enclave, do1, do2, ds, return_code)

    assert readers_of(enclave, RETURN_CODE, (ds, do1, do2)) == [ds, do1]


def test_one_owner_withholding_blocks_return_code():
    enclave, do1, do2, ds = build_quad()
    return_code = readable(RETURN_CODE, ds.email)
    submit(ds, enclave, do1, do2, CRASH_CODE, return_code)
    distribute(enclave, do1, do2)
    do1.approve_job(do1.jobs["j"], return_code)
    do2.approve_job(do2.jobs["j"], {})
    run_and_share(enclave, ds)

    assert not has_file(ds, RETURN_CODE)


def test_file_approved_by_some_parties_goes_to_no_one():
    """Each file has the approval of one party only, so nobody reads either."""
    enclave, do1, do2, ds = build_quad()
    parties = (ds, do1, do2)
    emails = [p.email for p in parties]
    submit(
        ds,
        enclave,
        do1,
        do2,
        CRASH_CODE,
        readable(["stderr.txt", RETURN_CODE], *emails),
    )
    distribute(enclave, do1, do2)
    do1.approve_job(do1.jobs["j"], readable(RETURN_CODE, *emails))
    do2.approve_job(do2.jobs["j"], readable("stderr.txt", *emails))
    run_and_share(enclave, *parties)

    assert readers_of(enclave, RETURN_CODE, parties) == []
    assert readers_of(enclave, "stderr.txt", parties) == []
    assert not any(has_file(p, RETURN_CODE) for p in parties)


@ENTRY_POINTS
def test_later_approval_releases_withheld_file(via_job):
    """A party can approve a file after the run, without a new submission."""
    enclave, do1, do2, ds = build_quad()
    logs = readable(LOGS, ds.email)
    submit(ds, enclave, do1, do2, OK_CODE, logs)
    run_to_completion(enclave, do1, do2, ds, {})
    assert not has_file(ds, "stdout.txt")

    # Both owners amend their approval, then the enclave applies it.
    for do in (do1, do2):
        assert update(do, logs, via_job) == logs
        assert do.jobs["j"].approved_read_permissions == logs
    run_and_share(enclave, ds)

    assert has_file(ds, "stdout.txt")


def test_later_approval_releases_outputs():
    """Outputs approved after the run reach the reader, as other files do."""
    enclave, do1, do2, ds = build_quad()
    outputs = readable(OUTPUTS, do1.email)
    submit(ds, enclave, do1, do2, OK_CODE, outputs)
    run_to_completion(enclave, do1, do2, ds, {})
    assert readers_of(enclave, RESULT, (do1, do2)) == []

    for do in (do1, do2):
        do.update_read_permissions(do.jobs["j"], outputs)
    run_and_share(enclave, do1)

    assert readers_of(enclave, RESULT, (do1, do2)) == [do1]
    assert has_file(do1, RESULT)


def test_unapproved_file_unreadable_during_sync(monkeypatch):
    """An unapproved file is never readable, whenever a sync runs.

    The job runner writes the logs into the review folder while the job runs,
    so only the permissions protect them. This test fails if a later change
    grants read on a file that not every party approved.
    """
    enclave, do1, do2, ds = build_quad()
    submit(ds, enclave, do1, do2, OK_CODE, readable(LOGS, ds.email))
    distribute(enclave, do1, do2)
    do1.approve_job(do1.jobs["j"], {})
    do2.approve_job(do2.jobs["j"], {})
    enclave.sync()

    exposed = []
    engine = enclave._rds.sync_engine
    real_sync = type(engine).sync

    def spy(self, *args, **kwargs):
        # readers_of reads the job list without a sync, so this cannot recurse.
        if self is engine:
            exposed.extend(
                (name, party.email)
                for name in LOGS
                for party in readers_of(enclave, name, (ds, do1, do2))
            )
        return real_sync(self, *args, **kwargs)

    monkeypatch.setenv("PRE_SYNC", "true")
    monkeypatch.setattr(type(engine), "sync", spy)
    enclave.run_jobs()

    assert not exposed, f"unapproved logs were readable during a sync: {exposed}"


def test_amendment_needs_approval_first():
    enclave, do1, do2, ds = build_quad()
    logs = readable(LOGS, ds.email)
    submit(ds, enclave, do1, do2, OK_CODE, logs)
    distribute(enclave, do1)

    with pytest.raises(ValueError, match="Approve the job first"):
        do1.update_read_permissions(do1.jobs["j"], logs)


def test_late_approval_reaches_named_data_owners():
    """distribute_results runs once, so a later approval needs its own path."""
    enclave, do1, do2, ds = build_quad()
    return_code = readable(RETURN_CODE, ds.email, do1.email, do2.email)
    submit(ds, enclave, do1, do2, CRASH_CODE, return_code)
    run_to_completion(enclave, do1, do2, ds, {})
    do1.sync()
    assert not has_file(do1, RETURN_CODE)

    for do in (do1, do2):
        do.update_read_permissions(do.jobs["j"], return_code)
    run_and_share(enclave, ds, do1, do2)

    assert has_file(ds, RETURN_CODE)
    assert has_file(do1, RETURN_CODE)
    assert has_file(do2, RETURN_CODE)


def test_approve_job_rejects_non_enclave_job():
    """JobInfo.approve takes a reason first, so read permissions would land there.

    SyftEnclaveClient.jobs wraps only a job whose job_type header says enclave.
    An unwrapped job used to record the read permissions as its approval reason
    and approve none, with no error.
    """
    enclave, do1, do2, ds = build_quad()
    logs = readable(LOGS, ds.email)
    submit(ds, enclave, do1, do2, OK_CODE, logs)
    distribute(enclave, do1)

    # The same job, read through the plain client, is a JobInfo and not an
    # EnclaveJobInfo.
    plain = do1._rds.job_client.jobs["j"]
    assert not isinstance(plain, EnclaveJobInfo)

    with pytest.raises(TypeError, match="not an enclave job"):
        do1.approve_job(plain, logs)


# -- the submitter's share flags, and the outputs ------------------------------


@pytest.mark.parametrize(
    "flags, logs_to, outputs_to",
    [
        (
            {"share_logs_with_submitter": True, "share_outputs_with_owners": True},
            "ds",
            "owners",
        ),
        (
            {"share_outputs_with_submitter": True, "share_logs_with_owners": True},
            "owners",
            "ds",
        ),
    ],
)
def test_share_flags_build_request(flags, logs_to, outputs_to):
    """The submitter flags name the submitter, the owner flags every data owner."""
    enclave, do1, do2, ds = build_quad()
    readers = {"ds": [ds.email], "owners": [do1.email, do2.email]}
    submit(ds, enclave, do1, do2, OK_CODE, **flags)
    distribute(enclave)

    expected = {
        **readable(LOGS, *readers[logs_to]),
        **readable(OUTPUTS, *readers[outputs_to]),
    }
    assert enclave.jobs["j"].requested_read_permissions == expected


def test_share_files_with_flag_raises():
    enclave, do1, do2, ds = SyftEnclaveClient.quad_with_mock_drive_service_connection(
        use_in_memory_cache=False
    )
    with pytest.raises(ValueError, match="share_files or the share flags"):
        submit(
            ds,
            enclave,
            do1,
            do2,
            OK_CODE,
            readable(LOGS, ds.email),
            share_logs_with_submitter=True,
        )


def test_outputs_reach_owner_only_after_every_party_approved():
    """do2 approves the outputs for itself only, so do1 gets none.

    The outputs go out in distribute_results, with the receipt, not after the run.
    """
    enclave, do1, do2, ds = build_quad()
    submit(ds, enclave, do1, do2, OK_CODE, share_outputs_with_owners=True)
    distribute(enclave, do1, do2)
    do1.approve_job(do1.jobs["j"])
    do2.approve_job(do2.jobs["j"], readable(OUTPUTS, do2.email))
    enclave.sync()
    enclave.run_jobs()
    assert readers_of(enclave, RESULT, (do1, do2)) == []

    enclave.distribute_results()
    do1.sync()
    do2.sync()
    assert readers_of(enclave, RESULT, (do1, do2)) == [do2]
    assert has_file(do2, RESULT)
    assert not has_file(do1, RESULT)


# -- the job-level calls, same as on a datasite ---------------------------------


def test_approve_signature_matches_datasite_job():
    """The DO approves both kinds of job with the same call."""
    for name in ("approve", "update_read_permissions"):
        base = inspect.signature(getattr(JobInfo, name))
        enclave = inspect.signature(getattr(EnclaveJobInfo, name))
        assert list(base.parameters) == list(enclave.parameters)


def test_data_owner_cannot_read_combined_approval():
    """A data owner holds only its own approval file."""
    enclave, do1, do2, ds = build_quad()
    logs = readable(LOGS, ds.email)
    submit(ds, enclave, do1, do2, OK_CODE, logs)
    distribute(enclave, do1)
    do1.jobs["j"].approve(share_files=logs)

    job = do1.jobs["j"]
    assert job.approved_read_permissions == logs
    with pytest.raises(LookupError, match="Only the enclave"):
        job.read_permissions_to_share


# -- warning and display --------------------------------------------------------


def test_owner_logs_approval_warns():
    enclave, do1, do2, ds = build_quad()
    logs = readable(LOGS, ds.email)
    submit(ds, enclave, do1, do2, OK_CODE, logs)
    distribute(enclave, do1)

    with pytest.warns(UserWarning, match="stdout or stderr"):
        do1.jobs["j"].approve(share_files=logs)


def test_display_on_enclave_shows_every_party():
    enclave, do1, do2, ds = build_quad()
    submit(
        ds, enclave, do1, do2, OK_CODE, readable(["stdout.txt", RETURN_CODE], ds.email)
    )
    distribute(enclave, do1)
    do1.jobs["j"].approve(share_files=readable(RETURN_CODE, ds.email))
    enclave.sync()

    rows = dict(enclave.jobs["j"].read_permission_rows())
    # The request survives the distribution, which writes the approval files.
    assert rows["Requested"] == (
        f"returncode.txt ({ds.email}), stdout.txt ({ds.email})"
    )
    assert rows[f"Approved by {do1.email}"] == f"returncode.txt ({ds.email})"
    assert rows[f"Approved by {do2.email}"] == "(pending)"
    assert rows["Approved by all"] == "none"


def test_display_on_data_owner_shows_own_approval():
    enclave, do1, do2, ds = build_quad()
    return_code = readable(RETURN_CODE, ds.email)
    submit(ds, enclave, do1, do2, OK_CODE, return_code)
    distribute(enclave, do1)
    do1.jobs["j"].approve(share_files=return_code)

    job = do1.jobs["j"]
    assert job.read_permission_rows() == [
        ("Requested", f"returncode.txt ({ds.email})"),
        ("Approved by you", f"returncode.txt ({ds.email})"),
        ("Approved by all", "needs the approval of every party"),
    ]
    assert "Approved by you:" in job._repr_html_()


# -- the request is DS-writable after the approval ------------------------------


def test_request_edit_after_approval_adds_nothing():
    """A wider approval than the request must not share the extra files later."""
    enclave, do1, do2, ds = build_quad()
    return_code = readable(RETURN_CODE, ds.email)
    wider = readable([*LOGS, RETURN_CODE], ds.email)
    submit(ds, enclave, do1, do2, OK_CODE, return_code)
    distribute(enclave, do1, do2)
    for do in (do1, do2):
        do.jobs["j"].approve(share_files=wider)
        assert do.jobs["j"].approved_read_permissions == return_code
    enclave.sync()

    # The DS widens its request on the enclave's copy after the approval.
    path = Path(enclave._rds.job_client.jobs["j"].job_submission_path) / "config.yaml"
    data = yaml.safe_load(path.read_text())
    data["headers"][REQUESTED_READ_PERMISSIONS_HEADER] = wider
    path.write_text(yaml.safe_dump(data))

    enclave.run_jobs()
    enclave.distribute_results()
    ds.sync()

    assert enclave.jobs["j"].read_permissions_to_share == return_code
    assert not has_file(ds, "stdout.txt")


def test_enclave_approve_refuses_misplaced_read_permissions():
    """Read permissions given as the reason, or as an item list, are refused."""
    enclave, do1, do2, ds = build_quad()
    logs = readable(LOGS, ds.email)
    submit(ds, enclave, do1, do2, OK_CODE, logs)
    distribute(enclave, do1)

    with pytest.raises(TypeError, match="share_files="):
        do1.jobs["j"].approve(logs)
    with pytest.raises(TypeError, match="dict of file names"):
        do1.jobs["j"].approve(share_files=["logs"])


# -- every required party, as the approval gate counts them ---------------------


def test_deleted_approval_file_blocks_late_approval():
    """The late-approval path must count every configured party, not the files."""
    enclave, do1, do2, ds = build_quad()
    logs = readable(LOGS, ds.email)
    submit(ds, enclave, do1, do2, OK_CODE, logs)
    run_to_completion(enclave, do1, do2, ds, {})

    do1.jobs["j"].update_read_permissions(logs)
    enclave.sync()
    review = Path(enclave._rds.job_client.jobs["j"].job_review_path)
    (review / enclave_approval_file_name(do2.email)).unlink()
    enclave.run_jobs()
    enclave.distribute_results()
    ds.sync()

    assert enclave.read_permissions_to_share(enclave._rds.job_client.jobs["j"]) == {}
    assert not has_file(ds, "stdout.txt")


def test_display_lists_party_without_file():
    enclave, do1, do2, ds = build_quad()
    submit(ds, enclave, do1, do2, OK_CODE, readable(LOGS, ds.email))
    distribute(enclave)
    review = Path(enclave._rds.job_client.jobs["j"].job_review_path)
    (review / enclave_approval_file_name(do2.email)).unlink()

    rows = dict(enclave.jobs["j"].read_permission_rows())
    assert rows[f"Approved by {do1.email}"] == "(pending)"
    assert rows[f"Approved by {do2.email}"] == "(no approval file)"


def test_job_reject_reaches_enclave():
    """``job.reject`` on a job from ``jobs`` is pushed like an approval."""
    enclave, do1, do2, ds = build_quad()
    logs = readable(LOGS, ds.email)
    submit(ds, enclave, do1, do2, OK_CODE, logs)
    distribute(enclave, do1, do2)
    do1.jobs["j"].approve(share_files=logs)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        do2.jobs["j"].reject("too much data")
    enclave.sync()

    assert enclave.jobs["j"].status == "rejected"
    assert not [w for w in record if str(w.message) == LOGS_WARNING]


# -- log requests after the run ---------------------------------------------------


PARTIES = ["do1@x.com", "do2@x.com"]


def write_log_request(review_dir, party, log_file="stderr.txt", **request):
    logs = {log_file: LogRequest(**request)}
    update = RequestedPermissionsUpdate(party=party, logs=logs)
    update.save_json(review_dir / log_request_file_name(party))


def test_log_shared_on_same_request_from_every_party(tmp_path):
    for party in PARTIES:
        write_log_request(tmp_path, party, readers=[DS], n_chars=10)
    assert logs_requested_by_all_parties(tmp_path, PARTIES) == {
        "stderr.txt": LogRequest(readers=[DS], n_chars=10)
    }


@pytest.mark.parametrize(
    "do2_request",
    [
        None,
        {"readers": [DS], "n_chars": 20},
        {"readers": [DS, "x@x.com"], "n_chars": 10},
    ],
    ids=["missing", "other_length", "other_readers"],
)
def test_log_request_differing_between_parties_shares_nothing(tmp_path, do2_request):
    write_log_request(tmp_path, "do1@x.com", readers=[DS], n_chars=10)
    if do2_request is not None:
        write_log_request(tmp_path, "do2@x.com", **do2_request)
    assert logs_requested_by_all_parties(tmp_path, PARTIES) == {}


def test_log_request_refuses_wildcard_reader():
    with pytest.raises(ValueError, match="Name each reader by email"):
        LogRequest(readers=["*"])


def test_log_request_for_other_file_is_ignored(tmp_path):
    write_log_request(tmp_path, "do1@x.com", log_file="outputs", readers=[DS])
    assert logs_requested_by_all_parties(tmp_path, ["do1@x.com"]) == {}


def run_crash_sharing_nothing():
    enclave, do1, do2, ds = build_quad()
    submit(ds, enclave, do1, do2, CRASH_CODE)
    run_to_completion(enclave, do1, do2, ds, {})
    return enclave, do1, do2, ds


def request_stderr(do, **request):
    with pytest.warns(UserWarning, match="stdout or stderr"):
        do.request_stderr(do.jobs["j"], **request)


def test_requested_stderr_head_reaches_submitter():
    enclave, do1, do2, ds = run_crash_sharing_nothing()
    assert not has_file(ds, "stderr.txt")

    for do in (do1, do2):
        request_stderr(do, readers=ds.email, n_chars=10)
    run_and_share(enclave, ds)

    enclave_review = Path(enclave._rds.job_client.jobs["j"].job_review_path)
    full = (enclave_review / "stderr.txt").read_text()
    assert has_file(ds, "stderr_10.txt")
    assert not has_file(ds, "stderr.txt")
    assert str(ds.jobs["j"].stderr) == full[:10]
    assert ds.jobs["j"].approved_read_permissions == {"stderr_10.txt": [ds.email]}


def test_one_owner_requesting_log_shares_nothing():
    enclave, do1, do2, ds = run_crash_sharing_nothing()
    request_stderr(do1, readers=ds.email)
    run_and_share(enclave, ds)
    assert not has_file(ds, "stderr.txt")


def test_log_request_defaults_to_data_owners():
    enclave, do1, do2, ds = run_crash_sharing_nothing()
    for do in (do1, do2):
        request_stderr(do)
    run_and_share(enclave, ds, do1, do2)

    assert readers_of(enclave, "stderr.txt", (ds, do1, do2)) == [do1, do2]
    assert has_file(do1, "stderr.txt")
    assert not has_file(ds, "stderr.txt")


def test_default_log_readers_ignore_owners_named_by_submitter():
    """The DS names the datasets, so naming itself there must not make it a reader."""
    enclave, do1, do2, ds = build_quad()
    datasets = {do1.email: ["dataset1"], do2.email: ["dataset2"], ds.email: ["x"]}
    submit(ds, enclave, do1, do2, CRASH_CODE, datasets=datasets)
    run_to_completion(enclave, do1, do2, ds, {})
    for do in (do1, do2):
        request_stderr(do)
    run_and_share(enclave, ds)

    assert readers_of(enclave, "stderr.txt", (ds, do1, do2)) == [do1, do2]
    assert not has_file(ds, "stderr.txt")


def test_submitter_cannot_request_logs():
    enclave, do1, do2, ds = run_crash_sharing_nothing()
    with pytest.raises(PermissionError, match="Only a data owner"):
        ds.request_stderr(ds.jobs["j"])
