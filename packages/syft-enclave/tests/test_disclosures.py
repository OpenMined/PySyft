"""Per-item disclosure: the enclave releases an item only under full agreement."""

import inspect
import json
import os
import random
import tempfile
from pathlib import Path

import pytest
import yaml

os.environ["PRE_SYNC"] = "false"

from syft_job.job import JobInfo  # noqa: E402
from syft_job.models import JobState, JobStatus  # noqa: E402
from syft_job.traceback_capture import FRAMES_FILENAME  # noqa: E402

from syft_enclaves import SyftEnclaveClient  # noqa: E402
from syft_enclaves.enclave_job_info import (  # noqa: E402
    DisclosureItem,
    EnclaveJobInfo,
    PartyApprovalStatus,
    approved_disclosures,
    enclave_approval_file_name,
    normalize_disclosures,
)

LOGS = DisclosureItem.LOGS.value
FRAMES = DisclosureItem.TRACEBACK_FRAMES.value
RETURN_CODE = DisclosureItem.RETURN_CODE.value


def write_approval(review_dir, party, status, disclosures):
    approval = PartyApprovalStatus(
        party=party, status=status, disclosures=disclosures or {}
    )
    approval.save_json(review_dir / enclave_approval_file_name(party))


# -- the resolution rule --------------------------------------------------------


def test_item_needs_every_party(tmp_path):
    write_approval(tmp_path, "do1@x.com", JobStatus.APPROVED, {LOGS: True})
    write_approval(tmp_path, "do2@x.com", JobStatus.APPROVED, {LOGS: True})
    assert approved_disclosures(tmp_path) == {LOGS}


def test_one_party_withholding_blocks_item(tmp_path):
    write_approval(tmp_path, "do1@x.com", JobStatus.APPROVED, {LOGS: True})
    write_approval(tmp_path, "do2@x.com", JobStatus.APPROVED, {FRAMES: True})
    assert approved_disclosures(tmp_path) == set()


def test_unapproved_party_blocks_everything(tmp_path):
    write_approval(tmp_path, "do1@x.com", JobStatus.APPROVED, {LOGS: True})
    write_approval(tmp_path, "do2@x.com", JobStatus.PENDING, {LOGS: True})
    assert approved_disclosures(tmp_path) == set()


def test_approval_file_without_disclosures_grants_nothing(tmp_path):
    """An older client writes no disclosures key, so it releases nothing."""
    path = tmp_path / enclave_approval_file_name("do1@x.com")
    path.write_text(json.dumps({"party": "do1@x.com", "status": "approved"}))
    assert PartyApprovalStatus.load_json(path).disclosures == {}
    assert approved_disclosures(tmp_path) == set()


def test_requested_set_narrows_result(tmp_path):
    write_approval(
        tmp_path, "do1@x.com", JobStatus.APPROVED, {LOGS: True, FRAMES: True}
    )
    assert approved_disclosures(tmp_path, [FRAMES]) == {FRAMES}


def test_no_approval_file_grants_nothing(tmp_path):
    assert approved_disclosures(tmp_path) == set()


def test_unknown_item_never_survives(tmp_path):
    write_approval(tmp_path, "do1@x.com", JobStatus.APPROVED, {"everything": True})
    assert approved_disclosures(tmp_path) == set()
    assert normalize_disclosures(["everything", LOGS]) == {LOGS: True}


def test_false_mapping_value_drops_item(tmp_path):
    """update_disclosures returns a map, so a caller can send one back.

    Every element used to count as a grant, therefore a map carrying False
    re-granted the item it was meant to drop.
    """
    assert normalize_disclosures({LOGS: True, FRAMES: False}) == {LOGS: True}

    granted = normalize_disclosures([LOGS])
    assert normalize_disclosures({**granted, LOGS: False}) == {}


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


def submit(ds, enclave, do1, do2, code, requested):
    path = Path(tempfile.mkdtemp()) / "main.py"
    path.write_text(code)
    ds.submit_python_job(
        enclave.email,
        str(path),
        "j",
        datasets={do1.email: ["dataset1"], do2.email: ["dataset2"]},
        request_disclosures=requested,
    )


OK_CODE = "import os, json\nos.makedirs('outputs', exist_ok=True)\nopen('outputs/r.json','w').write('{}')\n"
CRASH_CODE = "x = 1\ny = 2\nraise ValueError('account 88213 holds 4120550')\n"


# The DO calls the job, or the client method that wraps it.
ENTRY_POINTS = pytest.mark.parametrize("via_job", [False, True], ids=["client", "job"])


def approve(do, grants, via_job=False):
    if via_job:
        do.jobs["j"].approve(disclosures=grants)
    else:
        do.approve_job(do.jobs["j"], grants)


def update(do, grants, via_job=False):
    if via_job:
        return do.jobs["j"].update_disclosures(grants)
    return do.update_disclosures(do.jobs["j"], grants)


def run_to_completion(enclave, do1, do2, ds, grants, via_job=False):
    enclave.sync()
    enclave.receive_jobs()
    do1.sync()
    do2.sync()
    approve(do1, grants, via_job)
    approve(do2, grants, via_job)
    enclave.sync()
    enclave.run_jobs()
    enclave.distribute_results()
    ds.sync()


def test_submitter_gets_no_logs_without_grant():
    enclave, do1, do2, ds = build_quad()
    submit(ds, enclave, do1, do2, OK_CODE, [LOGS])
    run_to_completion(enclave, do1, do2, ds, None)

    review = Path(ds.jobs["j"].job_review_path)
    assert not (review / "stdout.txt").exists()
    assert not (review / "stderr.txt").exists()


@ENTRY_POINTS
def test_full_grant_sends_logs_to_submitter(via_job):
    enclave, do1, do2, ds = build_quad()
    submit(ds, enclave, do1, do2, OK_CODE, [LOGS])
    run_to_completion(enclave, do1, do2, ds, [LOGS], via_job)

    assert enclave.granted_disclosures(enclave.jobs["j"]) == {LOGS}
    assert (Path(ds.jobs["j"].job_review_path) / "stdout.txt").exists()


def test_submitter_gets_no_return_code_without_grant():
    enclave, do1, do2, ds = build_quad()
    submit(ds, enclave, do1, do2, CRASH_CODE, [RETURN_CODE])
    run_to_completion(enclave, do1, do2, ds, [LOGS])

    review = Path(ds.jobs["j"].job_review_path)
    assert not (review / "returncode.txt").exists()
    state = JobState.load(review / "state.yaml")
    assert state.status == JobStatus.FAILED
    assert state.return_code is None


def test_return_code_grant_sends_only_exit_code():
    enclave, do1, do2, ds = build_quad()
    submit(ds, enclave, do1, do2, CRASH_CODE, [RETURN_CODE])
    run_to_completion(enclave, do1, do2, ds, [RETURN_CODE])

    review = Path(ds.jobs["j"].job_review_path)
    assert (review / "returncode.txt").read_text().strip() == "1"
    assert not (review / "stderr.txt").exists()


def test_granted_frames_carry_position_not_message():
    enclave, do1, do2, ds = build_quad()
    submit(ds, enclave, do1, do2, CRASH_CODE, [FRAMES])
    run_to_completion(enclave, do1, do2, ds, [FRAMES])

    record = json.loads(
        (Path(ds.jobs["j"].job_review_path) / FRAMES_FILENAME).read_text()
    )
    entry = record["chain"][0]
    assert entry["type"] == "ValueError"
    assert {"file": "main.py", "line": 3} in entry["frames"]
    assert "4120550" not in json.dumps(record)


def test_frames_reach_data_owners():
    """A party that releases an item also receives it."""
    enclave, do1, do2, ds = build_quad()
    submit(ds, enclave, do1, do2, CRASH_CODE, [FRAMES])
    run_to_completion(enclave, do1, do2, ds, [FRAMES])
    do1.sync()

    assert (Path(do1.jobs["j"].job_review_path) / FRAMES_FILENAME).exists()


def test_one_owner_withholding_blocks_frames():
    enclave, do1, do2, ds = build_quad()
    submit(ds, enclave, do1, do2, CRASH_CODE, [FRAMES])
    enclave.sync()
    enclave.receive_jobs()
    do1.sync()
    do2.sync()
    do1.approve_job(do1.jobs["j"], [FRAMES])
    do2.approve_job(do2.jobs["j"], None)
    enclave.sync()
    enclave.run_jobs()
    enclave.distribute_results()
    ds.sync()

    assert not (Path(ds.jobs["j"].job_review_path) / FRAMES_FILENAME).exists()


@ENTRY_POINTS
def test_later_grant_releases_withheld_artifact(via_job):
    """A party can release an item after the run, without a new submission."""
    enclave, do1, do2, ds = build_quad()
    submit(ds, enclave, do1, do2, OK_CODE, [LOGS])
    run_to_completion(enclave, do1, do2, ds, None)
    assert not (Path(ds.jobs["j"].job_review_path) / "stdout.txt").exists()

    # Both owners amend their approval, then the enclave applies the grants.
    for do in (do1, do2):
        assert update(do, [LOGS], via_job) == {LOGS: True}
        assert do.jobs["j"].disclosures == {LOGS: True}
    enclave.sync()
    enclave.run_jobs()
    enclave.distribute_results()
    ds.sync()

    assert (Path(ds.jobs["j"].job_review_path) / "stdout.txt").exists()


def test_no_sync_while_ungranted_artifact_in_review(monkeypatch):
    """An ungranted artifact is never readable, whenever a sync runs.

    The job runner writes the logs into staging, so no ordering rule protects
    them. This test guards that property: it fails if a later change writes a
    gated artifact into the review folder before a release.
    """
    enclave, do1, do2, ds = build_quad()
    submit(ds, enclave, do1, do2, OK_CODE, [LOGS])
    enclave.sync()
    enclave.receive_jobs()
    do1.sync()
    do2.sync()
    do1.approve_job(do1.jobs["j"], None)
    do2.approve_job(do2.jobs["j"], None)
    enclave.sync()

    # Read the review path without a sync, so the spy below cannot recurse.
    review = Path(enclave._rds.job_client.jobs["j"].job_review_path)
    exposed = []
    engine = enclave._rds.sync_engine
    real_sync = type(engine).sync

    def spy(self, *args, **kwargs):
        if self is engine:
            exposed.append(
                sorted(
                    name
                    for name in ("stdout.txt", "stderr.txt")
                    if (review / name).exists()
                )
            )
        return real_sync(self, *args, **kwargs)

    monkeypatch.setenv("PRE_SYNC", "true")
    monkeypatch.setattr(type(engine), "sync", spy)
    enclave.run_jobs()

    assert not any(exposed), f"ungranted logs were readable during a sync: {exposed}"


def test_amendment_needs_approval_first():
    enclave, do1, do2, ds = build_quad()
    submit(ds, enclave, do1, do2, OK_CODE, [LOGS])
    enclave.sync()
    enclave.receive_jobs()
    do1.sync()

    with pytest.raises(ValueError, match="Approve the job first"):
        do1.update_disclosures(do1.jobs["j"], [LOGS])


def test_late_grant_reaches_data_owners():
    """distribute_results runs once, so a later release needs its own path."""
    enclave, do1, do2, ds = build_quad()
    submit(ds, enclave, do1, do2, CRASH_CODE, [FRAMES])
    run_to_completion(enclave, do1, do2, ds, None)
    do1.sync()
    assert not (Path(do1.jobs["j"].job_review_path) / FRAMES_FILENAME).exists()

    for do in (do1, do2):
        do.update_disclosures(do.jobs["j"], [FRAMES])
    enclave.sync()
    enclave.run_jobs()
    enclave.distribute_results()
    ds.sync()
    do1.sync()
    do2.sync()

    assert (Path(ds.jobs["j"].job_review_path) / FRAMES_FILENAME).exists()
    assert (Path(do1.jobs["j"].job_review_path) / FRAMES_FILENAME).exists()
    assert (Path(do2.jobs["j"].job_review_path) / FRAMES_FILENAME).exists()


def test_released_artifact_not_sent_twice():
    enclave, do1, do2, ds = build_quad()
    submit(ds, enclave, do1, do2, CRASH_CODE, [FRAMES])
    run_to_completion(enclave, do1, do2, ds, [FRAMES])

    job = enclave.jobs["j"]
    assert enclave._forward_new_releases(job) == []


def test_approve_job_rejects_non_enclave_job():
    """JobInfo.approve takes a reason first, so disclosures would land there.

    SyftEnclaveClient.jobs wraps only a job whose job_type header says enclave.
    An unwrapped job used to record the disclosures as its approval reason and
    grant nothing, with no error.
    """
    enclave, do1, do2, ds = build_quad()
    submit(ds, enclave, do1, do2, OK_CODE, [LOGS])
    enclave.sync()
    enclave.receive_jobs()
    do1.sync()

    # The same job, read through the plain client, is a JobInfo and not an
    # EnclaveJobInfo.
    plain = do1._rds.job_client.jobs["j"]
    assert not isinstance(plain, EnclaveJobInfo)

    with pytest.raises(TypeError, match="not an enclave job"):
        do1.approve_job(plain, [LOGS])


# -- the job-level calls, same as on a datasite ---------------------------------


def test_approve_signature_matches_datasite_job():
    """The DO approves both kinds of job with the same call."""
    for name in ("approve", "update_disclosures"):
        base = inspect.signature(getattr(JobInfo, name))
        enclave = inspect.signature(getattr(EnclaveJobInfo, name))
        assert list(base.parameters) == list(enclave.parameters)


def test_data_owner_cannot_read_combined_grant():
    """A data owner holds only its own approval file."""
    enclave, do1, do2, ds = build_quad()
    submit(ds, enclave, do1, do2, OK_CODE, [LOGS])
    enclave.sync()
    enclave.receive_jobs()
    do1.sync()
    do1.jobs["j"].approve(disclosures=[LOGS])

    job = do1.jobs["j"]
    assert job.disclosures == {LOGS: True}
    with pytest.raises(LookupError, match="Only the enclave"):
        job.granted_disclosures


# -- warning and display --------------------------------------------------------


def test_owner_logs_grant_warns():
    enclave, do1, do2, ds = build_quad()
    submit(ds, enclave, do1, do2, OK_CODE, [LOGS])
    enclave.sync()
    enclave.receive_jobs()
    do1.sync()

    with pytest.warns(UserWarning, match="stdout and stderr"):
        do1.jobs["j"].approve(disclosures=[LOGS])


def test_display_on_enclave_shows_every_party():
    enclave, do1, do2, ds = build_quad()
    submit(ds, enclave, do1, do2, OK_CODE, [LOGS, FRAMES])
    enclave.sync()
    enclave.receive_jobs()
    do1.sync()
    do1.jobs["j"].approve(disclosures=[FRAMES])
    enclave.sync()

    rows = dict(enclave.jobs["j"].disclosure_rows())
    assert rows["Requested"] == "logs, traceback_frames"
    assert rows[f"Granted by {do1.email}"] == "traceback_frames"
    assert rows[f"Granted by {do2.email}"] == "(pending)"
    assert rows["To submitter"] == "none"


def test_display_on_data_owner_shows_own_grant():
    enclave, do1, do2, ds = build_quad()
    submit(ds, enclave, do1, do2, OK_CODE, [FRAMES])
    enclave.sync()
    enclave.receive_jobs()
    do1.sync()
    do1.jobs["j"].approve(disclosures=[FRAMES])

    job = do1.jobs["j"]
    assert job.disclosure_rows() == [
        ("Requested", "traceback_frames"),
        ("Your grant", "traceback_frames"),
        ("To submitter", "needs the grant of every party"),
    ]
    assert "Your grant:" in job._repr_html_()


# -- the request is DS-writable after the approval ------------------------------


def test_request_edit_after_approval_adds_nothing():
    """A wider grant than the request must not release the extra items later."""
    enclave, do1, do2, ds = build_quad()
    submit(ds, enclave, do1, do2, OK_CODE, [FRAMES])
    enclave.sync()
    enclave.receive_jobs()
    do1.sync()
    do2.sync()
    for do in (do1, do2):
        do.jobs["j"].approve(disclosures=[LOGS, FRAMES])
        assert do.jobs["j"].disclosures == {FRAMES: True}
    enclave.sync()

    # The DS widens its request on the enclave's copy after the approval.
    path = Path(enclave._rds.job_client.jobs["j"].job_submission_path) / "config.yaml"
    data = yaml.safe_load(path.read_text())
    data["headers"]["requested_disclosures"] = [LOGS, FRAMES]
    path.write_text(yaml.safe_dump(data))

    enclave.run_jobs()
    enclave.distribute_results()
    ds.sync()

    assert enclave.jobs["j"].granted_disclosures == {FRAMES}
    assert not (Path(ds.jobs["j"].job_review_path) / "stdout.txt").exists()


def test_enclave_approve_refuses_items_as_reason():
    enclave, do1, do2, ds = build_quad()
    submit(ds, enclave, do1, do2, OK_CODE, [LOGS])
    enclave.sync()
    enclave.receive_jobs()
    do1.sync()

    with pytest.raises(TypeError, match="disclosures="):
        do1.jobs["j"].approve([LOGS])
