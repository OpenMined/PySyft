"""Per-item disclosure: the enclave releases an item only under full agreement."""

import json
import os
import random
import shutil
import tempfile
from pathlib import Path

import pytest

os.environ["PRE_SYNC"] = "false"

from syft_job.models import JobStatus  # noqa: E402
from syft_job.traceback_capture import FRAMES_FILENAME  # noqa: E402

from syft_enclaves import SyftEnclaveClient  # noqa: E402
from syft_enclaves.enclave_job_info import (  # noqa: E402
    DisclosureItem,
    PartyApprovalStatus,
    approved_disclosures,
    enclave_approval_file_name,
    normalize_disclosures,
)

LOGS = DisclosureItem.LOGS.value
FRAMES = DisclosureItem.TRACEBACK_FRAMES.value


def write_approval(review_dir, party, status, disclosures):
    approval = PartyApprovalStatus(
        party=party, status=status, disclosures=disclosures or {}
    )
    approval.save_json(review_dir / enclave_approval_file_name(party))


# -- the resolution rule --------------------------------------------------------


def test_an_item_needs_every_party(tmp_path):
    write_approval(tmp_path, "do1@x.com", JobStatus.APPROVED, {LOGS: True})
    write_approval(tmp_path, "do2@x.com", JobStatus.APPROVED, {LOGS: True})
    assert approved_disclosures(tmp_path) == {LOGS}


def test_one_party_withholding_blocks_the_item(tmp_path):
    write_approval(tmp_path, "do1@x.com", JobStatus.APPROVED, {LOGS: True})
    write_approval(tmp_path, "do2@x.com", JobStatus.APPROVED, {FRAMES: True})
    assert approved_disclosures(tmp_path) == set()


def test_a_party_that_has_not_approved_blocks_everything(tmp_path):
    write_approval(tmp_path, "do1@x.com", JobStatus.APPROVED, {LOGS: True})
    write_approval(tmp_path, "do2@x.com", JobStatus.PENDING, {LOGS: True})
    assert approved_disclosures(tmp_path) == set()


def test_an_approval_file_without_the_field_grants_nothing(tmp_path):
    """An older client writes no disclosures key, so it releases nothing."""
    path = tmp_path / enclave_approval_file_name("do1@x.com")
    path.write_text(json.dumps({"party": "do1@x.com", "status": "approved"}))
    assert PartyApprovalStatus.load_json(path).disclosures == {}
    assert approved_disclosures(tmp_path) == set()


def test_the_requested_set_narrows_the_result(tmp_path):
    write_approval(tmp_path, "do1@x.com", JobStatus.APPROVED, {LOGS: True, FRAMES: True})
    assert approved_disclosures(tmp_path, [FRAMES]) == {FRAMES}


def test_no_approval_file_grants_nothing(tmp_path):
    assert approved_disclosures(tmp_path) == set()


def test_an_unknown_item_never_survives(tmp_path):
    write_approval(tmp_path, "do1@x.com", JobStatus.APPROVED, {"everything": True})
    assert approved_disclosures(tmp_path) == set()
    assert normalize_disclosures(["everything", LOGS]) == {LOGS: True}


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
CRASH_CODE = "x = 1\ny = 2\nraise ValueError('patient 4171 is positive')\n"


def run_to_completion(enclave, do1, do2, ds, grants):
    enclave.sync()
    enclave.receive_jobs()
    do1.sync()
    do2.sync()
    do1.approve_job(do1.jobs["j"], grants)
    do2.approve_job(do2.jobs["j"], grants)
    enclave.sync()
    enclave.run_jobs()
    enclave.distribute_results()
    ds.sync()


def test_without_a_grant_the_submitter_gets_no_logs():
    enclave, do1, do2, ds = build_quad()
    submit(ds, enclave, do1, do2, OK_CODE, [LOGS])
    run_to_completion(enclave, do1, do2, ds, None)

    review = Path(ds.jobs["j"].job_review_path)
    assert not (review / "stdout.txt").exists()
    assert not (review / "stderr.txt").exists()


def test_a_full_grant_sends_the_logs_to_the_submitter():
    enclave, do1, do2, ds = build_quad()
    submit(ds, enclave, do1, do2, OK_CODE, [LOGS])
    run_to_completion(enclave, do1, do2, ds, [LOGS])

    assert enclave.granted_disclosures(enclave.jobs["j"]) == {LOGS}
    assert (Path(ds.jobs["j"].job_review_path) / "stdout.txt").exists()


def test_granted_frames_carry_the_position_but_not_the_message():
    enclave, do1, do2, ds = build_quad()
    submit(ds, enclave, do1, do2, CRASH_CODE, [FRAMES])
    run_to_completion(enclave, do1, do2, ds, [FRAMES])

    record = json.loads(
        (Path(ds.jobs["j"].job_review_path) / FRAMES_FILENAME).read_text()
    )
    entry = record["chain"][0]
    assert entry["type"] == "ValueError"
    assert {"file": "main.py", "line": 3} in entry["frames"]
    assert "positive" not in json.dumps(record)


def test_frames_reach_the_data_owners_too():
    """A party that releases an item also receives it."""
    enclave, do1, do2, ds = build_quad()
    submit(ds, enclave, do1, do2, CRASH_CODE, [FRAMES])
    run_to_completion(enclave, do1, do2, ds, [FRAMES])
    do1.sync()

    assert (Path(do1.jobs["j"].job_review_path) / FRAMES_FILENAME).exists()


def test_one_owner_withholding_blocks_the_frames():
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


def test_a_later_grant_releases_a_withheld_artifact():
    """A party can release an item after the run, without a new submission."""
    enclave, do1, do2, ds = build_quad()
    submit(ds, enclave, do1, do2, OK_CODE, [LOGS])
    run_to_completion(enclave, do1, do2, ds, None)
    assert not (Path(ds.jobs["j"].job_review_path) / "stdout.txt").exists()

    # Both owners amend their approval, then the enclave applies the grants.
    for do in (do1, do2):
        assert do.update_disclosures(do.jobs["j"], [LOGS]) == {LOGS: True}
    enclave.sync()
    enclave.run_jobs()
    enclave.distribute_results()
    ds.sync()

    assert (Path(ds.jobs["j"].job_review_path) / "stdout.txt").exists()


def test_no_sync_runs_while_an_ungranted_artifact_sits_in_review(monkeypatch):
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


def test_an_amendment_needs_an_approval_first():
    enclave, do1, do2, ds = build_quad()
    submit(ds, enclave, do1, do2, OK_CODE, [LOGS])
    enclave.sync()
    enclave.receive_jobs()
    do1.sync()

    with pytest.raises(ValueError, match="Approve the job first"):
        do1.update_disclosures(do1.jobs["j"], [LOGS])


def test_a_late_grant_reaches_the_data_owners_too():
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


def test_a_released_artifact_is_not_sent_twice():
    enclave, do1, do2, ds = build_quad()
    submit(ds, enclave, do1, do2, CRASH_CODE, [FRAMES])
    run_to_completion(enclave, do1, do2, ds, [FRAMES])

    job = enclave.jobs["j"]
    assert enclave._forward_new_releases(job) == []


def test_a_changed_record_reaches_the_parties_again():
    """The forwarded record holds a digest, so new content goes out again."""
    enclave, do1, do2, ds = build_quad()
    submit(ds, enclave, do1, do2, CRASH_CODE, [FRAMES])
    run_to_completion(enclave, do1, do2, ds, [FRAMES])

    review = Path(enclave.jobs["j"].job_review_path)
    first = json.loads((review / FRAMES_FILENAME).read_text())
    assert enclave._forward_new_releases(enclave.jobs["j"]) == []

    # A later run writes a different record under the same name.
    (review / FRAMES_FILENAME).write_text(
        json.dumps({"chain": [{"type": "KeyError", "frames": []}]})
    )
    assert enclave._forward_new_releases(enclave.jobs["j"]) == [FRAMES_FILENAME]

    do1.sync()
    delivered = json.loads(
        (Path(do1.jobs["j"].job_review_path) / FRAMES_FILENAME).read_text()
    )
    assert delivered["chain"][0]["type"] == "KeyError"
    assert delivered != first


def test_a_new_run_is_not_skipped_as_already_shared():
    """distribute_results marks a failed job, so a later run must not be skipped.

    The marker names the run it covers. An end-to-end rerun cannot drive this:
    `rerun()` leaves code/.venv and `uv venv` then fails, for every python job.
    """
    enclave, do1, do2, ds = build_quad()
    submit(ds, enclave, do1, do2, CRASH_CODE, [FRAMES])
    run_to_completion(enclave, do1, do2, ds, [FRAMES])

    job = enclave.jobs["j"]
    assert job.status == "failed"
    assert enclave._results_already_shared(job)

    # A later run writes a new state, which is what rerun() prepares for.
    state_file = Path(job.job_review_path) / "state.yaml"
    state_file.write_text(state_file.read_text() + "\n# next run\n")
    assert not enclave._results_already_shared(job)


def test_a_marker_from_an_older_client_still_counts_as_shared():
    enclave, do1, do2, ds = build_quad()
    submit(ds, enclave, do1, do2, OK_CODE, [LOGS])
    run_to_completion(enclave, do1, do2, ds, [LOGS])

    job = enclave.jobs["j"]
    marker = Path(job.job_review_path) / "results_shared"
    marker.write_text("shared")

    assert enclave._results_already_shared(job)
    # The marker now names the current run, so the next run redistributes.
    assert json.loads(marker.read_text())["state"] == enclave._state_digest(job)


# A job that fails once, then succeeds, with no edit between the runs.
FAIL_THEN_PASS_CODE = """\
import json, os

flag = {flag!r}
if not os.path.exists(flag):
    open(flag, "w").write("x")
    raise ValueError("patient 4171 is positive")

os.makedirs("outputs", exist_ok=True)
open("outputs/r.json", "w").write(json.dumps({{"ok": 1}}))
"""


def rerun_on_the_enclave(enclave):
    """Prepare a rerun, and clear what blocks one.

    `rerun()` leaves code/.venv, and `run.sh` then runs `uv venv`, which stops
    under `set -euo pipefail`. That defect predates the staging work and no
    caller hits it, so the test clears the directory itself.
    """
    job = enclave.jobs["j"]
    job.rerun()
    shutil.rmtree(Path(job.job_submission_path) / "code" / ".venv", ignore_errors=True)


def test_a_run_that_does_not_fail_replaces_the_crash_record():
    enclave, do1, do2, ds = build_quad()
    flag = str(Path(tempfile.mkdtemp()) / "once")
    submit(ds, enclave, do1, do2, FAIL_THEN_PASS_CODE.format(flag=flag), [FRAMES])
    run_to_completion(enclave, do1, do2, ds, [FRAMES])
    do1.sync()

    assert enclave.jobs["j"].status == "failed"
    do1_record = Path(do1.jobs["j"].job_review_path) / FRAMES_FILENAME
    assert json.loads(do1_record.read_text())["chain"][0]["type"] == "ValueError"

    rerun_on_the_enclave(enclave)
    enclave.run_jobs()
    enclave.distribute_results()
    ds.sync()
    do1.sync()

    assert enclave.jobs["j"].status == "done"
    # No party keeps a crash description for a run that did not fail.
    for holder in (ds, do1):
        record = Path(holder.jobs["j"].job_review_path) / FRAMES_FILENAME
        assert json.loads(record.read_text()) == {"chain": []}


def test_no_replacement_when_the_parties_hold_no_record():
    """A job that never failed sends nothing, so there is nothing to replace."""
    enclave, do1, do2, ds = build_quad()
    submit(ds, enclave, do1, do2, OK_CODE, [FRAMES])
    run_to_completion(enclave, do1, do2, ds, [FRAMES])

    job = enclave.jobs["j"]
    assert not (Path(job.job_review_path) / FRAMES_FILENAME).exists()
    assert enclave._forward_new_releases(job) == []
