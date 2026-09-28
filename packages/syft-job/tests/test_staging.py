"""Staged artifacts reach the submitter only when a release moves them."""

import json
import warnings
from pathlib import Path

import pytest
import yaml
from syft_perms import SyftPermContext

from syft_job.client import JobClient
from syft_job.config import SyftJobConfig
from syft_job.disclosures import LOGS_WARNING, DisclosureItem
from syft_job.job import JobInfo
from syft_job.job_runner import SyftJobRunner
from syft_job.models import JobState, JobStatus
from syft_job.traceback_capture import FRAMES_FILENAME

DO_EMAIL = "do@test.org"
DS_EMAIL = "ds@test.org"
LOG_FILES = ("stdout.txt", "stderr.txt", "returncode.txt")
LOGS = DisclosureItem.LOGS.value
FRAMES = DisclosureItem.TRACEBACK_FRAMES.value
RETURN_CODE = DisclosureItem.RETURN_CODE.value

OK_PY = """\
import os

print("hello from job")
os.makedirs("outputs", exist_ok=True)
with open("outputs/result.txt", "w") as f:
    f.write("done")
"""

EXIT_3_PY = """\
import sys

sys.exit(3)
"""

CRASH_PY = """\
x = 1
raise ValueError("account 88213 holds 4120550")
"""


def do_config_for(tmp_path: Path) -> SyftJobConfig:
    return SyftJobConfig(
        syftbox_folder=tmp_path / "SyftBox", current_user_email=DO_EMAIL
    )


def submit_only(
    tmp_path: Path, requested: list[str] | None = None, code: str = OK_PY
) -> JobInfo:
    """Submit one job and return it as the DO sees it, still pending."""
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir(exist_ok=True)
    code_file = tmp_path / "main.py"
    code_file.write_text(code)
    ds_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS_EMAIL)
    JobClient(config=ds_config).submit_python_job(
        user=DO_EMAIL,
        code_path=str(code_file),
        job_name="test.job",
        request_disclosures=requested,
    )
    return JobClient(config=do_config_for(tmp_path)).jobs[0]


def run_approved(tmp_path: Path, share_logs: bool | None = None) -> None:
    """Run the approved jobs as the DO. ``share_logs=None`` uses the default."""
    kwargs = {} if share_logs is None else {"share_logs_with_submitter": share_logs}
    SyftJobRunner(config=do_config_for(tmp_path)).process_approved_jobs(
        stream_output=False, timeout=180, **kwargs
    )


def run_job(
    tmp_path: Path,
    code: str,
    share_logs: bool | None,
    requested: list[str] | None = None,
    grants: list[str] | None = None,
):
    """Submit and run one job. Returns (job, review_dir, staging_dir).

    ``requested`` is what the DS asks for, and ``grants`` is what the DO
    releases at approval.
    """
    submit_only(tmp_path, requested, code).approve(disclosures=grants)
    run_approved(tmp_path, share_logs)
    do_config = do_config_for(tmp_path)
    return (
        JobClient(config=do_config).jobs[0],
        do_config.get_review_job_dir(DO_EMAIL, DS_EMAIL, "test.job"),
        do_config.get_staging_job_dir(DO_EMAIL, DS_EMAIL, "test.job"),
    )


def test_state_yaml_omits_exit_code(tmp_path):
    """The DS reads state.yaml, so the exact exit code must not be in it."""
    job, review, staging = run_job(tmp_path, EXIT_3_PY, share_logs=None)
    state = JobState.load(review / "state.yaml")
    assert state.status == JobStatus.FAILED
    assert state.return_code is None
    assert (staging / "returncode.txt").read_text().strip() == "3"


def test_failure_after_run_stages_return_code(tmp_path, monkeypatch):
    """The runner stages -1 when it fails after the job ran."""

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(SyftJobRunner, "_move_outputs_to_review", fail)
    job, review, staging = run_job(tmp_path, OK_PY, share_logs=None)

    assert (staging / "returncode.txt").read_text().strip() == "-1"
    assert not (review / "returncode.txt").exists()
    state = JobState.load(review / "state.yaml")
    assert state.status == JobStatus.FAILED
    assert state.return_code is None


def test_release_moves_logs_into_review(tmp_path):
    job, review, staging = run_job(tmp_path, OK_PY, share_logs=False)
    assert sorted(job.release_logs()) == sorted(LOG_FILES)
    assert (review / "stdout.txt").exists()
    assert (review / "returncode.txt").exists()
    assert not (staging / "stdout.txt").exists()


def test_default_holds_logs_back(tmp_path):
    """The DS reads a log only after the DO releases it."""
    job, review, staging = run_job(tmp_path, OK_PY, share_logs=None)
    for name in LOG_FILES:
        assert (staging / name).exists()
        assert not (review / name).exists()
    assert (staging / "returncode.txt").read_text().strip() == "0"


def test_share_logs_releases_logs(tmp_path):
    job, review, staging = run_job(tmp_path, OK_PY, share_logs=True)
    assert (review / "stdout.txt").exists()
    assert (review / "returncode.txt").exists()
    assert not (staging / "stdout.txt").exists()
    for name in LOG_FILES:
        assert ds_can_read(tmp_path, name)


def test_second_release_is_harmless(tmp_path):
    job, review, staging = run_job(tmp_path, OK_PY, share_logs=False)
    job.release_logs()
    assert job.release_logs() == []
    assert (review / "stdout.txt").exists()


def test_owner_reads_staged_log_through_viewer(tmp_path):
    job, review, staging = run_job(tmp_path, OK_PY, share_logs=False)
    assert "hello from job" in str(job.stdout)
    assert job.artifact_path("stdout.txt") == staging / "stdout.txt"
    job.release_logs()
    assert job.artifact_path("stdout.txt") == review / "stdout.txt"


def test_traceback_record_is_staged(tmp_path):
    job, review, staging = run_job(tmp_path, CRASH_PY, share_logs=False)
    record_path = staging / FRAMES_FILENAME
    assert record_path.exists()
    assert not (review / FRAMES_FILENAME).exists()

    record = json.loads(record_path.read_text())
    assert record["chain"][0]["type"] == "ValueError"
    assert "4120550" not in record_path.read_text()

    assert job.release_artifacts([FRAMES_FILENAME]) == [FRAMES_FILENAME]
    assert (review / FRAMES_FILENAME).exists()


def test_rerun_clears_both_locations(tmp_path):
    """A staged log from the old run must not survive into a later release."""
    job, review, staging = run_job(tmp_path, OK_PY, share_logs=False)
    (review / "returncode.txt").write_text("0")
    assert (staging / "stdout.txt").exists()

    job.rerun()

    assert not (staging / "stdout.txt").exists()
    assert not (staging / "stderr.txt").exists()
    assert not (review / "stdout.txt").exists()


def test_rerun_clears_crash_record(tmp_path):
    """A crash record from the old run must not survive a later release.

    A rerun that succeeds writes no record, so a leftover record would describe
    a failure that the new run never had.
    """
    job, review, staging = run_job(tmp_path, CRASH_PY, share_logs=False)
    assert (staging / FRAMES_FILENAME).exists()

    job.rerun()

    assert not (staging / FRAMES_FILENAME).exists()
    assert not (review / FRAMES_FILENAME).exists()


# -- disclosures: requested by the DS, granted by the DO ------------------------


def ds_can_read(tmp_path: Path, name: str) -> bool:
    ctx = SyftPermContext(datasite=tmp_path / "SyftBox" / DO_EMAIL)
    return ctx.open(
        f"app_data/job/review/{DS_EMAIL}/v1/test.job/{name}"
    ).has_read_access(DS_EMAIL)


def test_submission_records_known_requests_only(tmp_path):
    job, _, _ = run_job(tmp_path, OK_PY, None, requested=[LOGS, "everything"])
    assert job.requested_disclosures == [LOGS]


def test_release_is_requested_and_granted(tmp_path):
    """A granted item that the DS did not request stays in staging."""
    job, review, staging = run_job(
        tmp_path, CRASH_PY, None, requested=[LOGS], grants=[LOGS, FRAMES]
    )
    assert job.granted_disclosures == {LOGS}
    for name in ("stdout.txt", "stderr.txt"):
        assert (review / name).exists()
        assert ds_can_read(tmp_path, name)
    assert (staging / FRAMES_FILENAME).exists()
    assert (staging / "returncode.txt").exists()


def test_request_without_grant_releases_nothing(tmp_path):
    job, review, staging = run_job(
        tmp_path, CRASH_PY, None, requested=[LOGS, FRAMES, RETURN_CODE]
    )
    assert job.granted_disclosures == set()
    for name in (*LOG_FILES, FRAMES_FILENAME):
        assert not (review / name).exists()


def test_granted_frames_released_without_logs(tmp_path):
    job, review, staging = run_job(
        tmp_path, CRASH_PY, None, requested=[FRAMES], grants=[FRAMES]
    )
    assert (review / FRAMES_FILENAME).exists()
    assert not (review / "stderr.txt").exists()


def test_later_grant_releases_finished_job(tmp_path):
    job, review, staging = run_job(tmp_path, OK_PY, None, requested=[RETURN_CODE])
    assert not (review / "returncode.txt").exists()

    assert job.update_disclosures([RETURN_CODE]) == {RETURN_CODE: True}

    assert (review / "returncode.txt").read_text().strip() == "0"
    assert ds_can_read(tmp_path, "returncode.txt")
    assert not (review / "stdout.txt").exists()


def test_later_grant_through_stale_job_releases(tmp_path):
    """A job object read before the run still sees that the run finished."""
    job = submit_only(tmp_path, [LOGS])
    job.approve()
    run_approved(tmp_path)

    job.update_disclosures([LOGS])

    review = do_config_for(tmp_path).get_review_job_dir(DO_EMAIL, DS_EMAIL, "test.job")
    assert (review / "stdout.txt").exists()


def test_update_disclosures_needs_approval_first(tmp_path):
    job = submit_only(tmp_path)
    with pytest.raises(ValueError, match="Approve the job first"):
        job.update_disclosures([LOGS])


def test_grant_survives_rerun(tmp_path):
    job, review, staging = run_job(
        tmp_path, OK_PY, None, requested=[LOGS], grants=[LOGS]
    )
    job.rerun()
    assert job.disclosures == {LOGS: True}


# -- warning and display --------------------------------------------------------


def logs_warnings(record) -> list:
    return [w for w in record if str(w.message) == LOGS_WARNING]


def test_approve_warns_when_logs_go_out(tmp_path):
    job = submit_only(tmp_path, [LOGS])
    with pytest.warns(UserWarning, match="stdout and stderr") as record:
        job.approve(disclosures=[LOGS])
    # The warning points at the caller, not at the helper.
    assert logs_warnings(record)[0].filename == __file__


def test_unrequested_logs_grant_does_not_warn(tmp_path):
    """A grant of an item the DS did not request releases nothing."""
    job = submit_only(tmp_path, [FRAMES])
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        job.approve(disclosures=[LOGS, FRAMES])
    assert logs_warnings(record) == []


def test_update_disclosures_warns_when_logs_go_out(tmp_path):
    job = submit_only(tmp_path, [LOGS])
    job.approve()
    with pytest.warns(UserWarning, match="stdout and stderr"):
        job.update_disclosures([LOGS])


def test_display_shows_disclosure_state(tmp_path):
    job = submit_only(tmp_path, [LOGS, FRAMES])
    job.approve(disclosures=[FRAMES, RETURN_CODE])

    # Only the requested item is stored, so return_code is not granted.
    assert job.disclosure_rows() == [
        ("Requested", "logs, traceback_frames"),
        ("Granted", "traceback_frames"),
        ("To submitter", "traceback_frames"),
    ]
    assert "To submitter: traceback_frames" in str(job)
    html = job._repr_html_()
    assert "To submitter:" in html
    assert "logs, traceback_frames" in html


def test_display_without_disclosures(tmp_path):
    job = submit_only(tmp_path, None)
    assert "Requested: none; Granted: none; To submitter: none" in str(job)


# -- the request is DS-writable after the approval ------------------------------


def edit_request(job: JobInfo, requested: list[str]) -> None:
    """Rewrite the request in config.yaml, as a DS can after the approval."""
    path = job.job_submission_path / "config.yaml"
    data = yaml.safe_load(path.read_text())
    data["headers"]["requested_disclosures"] = requested
    path.write_text(yaml.safe_dump(data))


def test_grant_stores_only_requested_items(tmp_path):
    job = submit_only(tmp_path, [FRAMES])
    job.approve(disclosures=[LOGS, FRAMES])
    assert job.disclosures == {FRAMES: True}


def test_request_edit_after_approval_adds_nothing(tmp_path):
    """A wider grant than the request must not release the extra items later."""
    job = submit_only(tmp_path, [FRAMES])
    job.approve(disclosures=[LOGS, FRAMES])
    edit_request(job, [LOGS, FRAMES])

    run_approved(tmp_path)

    review = do_config_for(tmp_path).get_review_job_dir(DO_EMAIL, DS_EMAIL, "test.job")
    assert not (review / "stdout.txt").exists()
    assert not ds_can_read(tmp_path, "stdout.txt")


def test_approve_refuses_items_as_reason(tmp_path):
    """``approve(["logs"])`` would grant nothing and record the list as reason."""
    job = submit_only(tmp_path, [LOGS])
    with pytest.raises(TypeError, match="disclosures="):
        job.approve([LOGS])
    assert job.status == "pending"
