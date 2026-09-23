"""Staged artifacts reach the submitter only when a release moves them."""

import json
from pathlib import Path


from syft_job.client import JobClient
from syft_job.config import SyftJobConfig
from syft_job.job_runner import SyftJobRunner
from syft_job.models import JobState, JobStatus
from syft_job.traceback_capture import FRAMES_FILENAME

DO_EMAIL = "do@test.org"
DS_EMAIL = "ds@test.org"
LOG_FILES = ("stdout.txt", "stderr.txt", "returncode.txt")

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


def run_job(tmp_path: Path, code: str, share_logs: bool | None):
    """Submit and run one job. Returns (job, review_dir, staging_dir).

    ``share_logs=None`` uses the runner default.
    """
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir(exist_ok=True)
    code_file = tmp_path / "main.py"
    code_file.write_text(code)

    do_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DO_EMAIL)
    ds_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS_EMAIL)
    ds_client = JobClient(config=ds_config)
    do_client = JobClient(config=do_config)
    do_runner = SyftJobRunner(config=do_config)

    ds_client.submit_python_job(
        user=DO_EMAIL, code_path=str(code_file), job_name="test.job"
    )
    job = do_client.jobs[0]
    job.approve()
    kwargs = {} if share_logs is None else {"share_logs_with_submitter": share_logs}
    do_runner.process_approved_jobs(stream_output=False, timeout=180, **kwargs)
    job = do_client.jobs[0]
    return (
        job,
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
