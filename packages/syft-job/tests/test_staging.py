"""Staged artifacts reach the submitter only when a release moves them."""

import json
from pathlib import Path

import pytest

from syft_job.client import JobClient
from syft_job.config import SyftJobConfig
from syft_job.job_runner import SyftJobRunner
from syft_job.traceback_capture import (
    BUNDLE_FILENAME,
    FRAMES_FILENAME,
    load_or_create_bundle,
)

DO_EMAIL = "do@test.org"
DS_EMAIL = "ds@test.org"

OK_PY = """\
import os

print("hello from job")
os.makedirs("outputs", exist_ok=True)
with open("outputs/result.txt", "w") as f:
    f.write("done")
"""

CRASH_PY = """\
x = 1
raise ValueError("patient 4171 is positive")
"""


def run_job(tmp_path: Path, code: str, share_logs: bool):
    """Submit and run one job. Returns (job, review_dir, staging_dir)."""
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
    do_runner.process_approved_jobs(
        stream_output=False, timeout=180, share_logs_with_submitter=share_logs
    )
    job = do_client.jobs[0]
    return (
        job,
        do_config.get_review_job_dir(DO_EMAIL, DS_EMAIL, "test.job"),
        do_config.get_staging_job_dir(DO_EMAIL, DS_EMAIL, "test.job"),
    )


def test_logs_wait_in_staging_when_nothing_released_them(tmp_path):
    job, review, staging = run_job(tmp_path, OK_PY, share_logs=False)
    assert (staging / "stdout.txt").exists()
    assert (staging / "stderr.txt").exists()
    assert not (review / "stdout.txt").exists()
    assert not (review / "stderr.txt").exists()


def test_returncode_stays_in_review(tmp_path):
    """returncode.txt holds one integer, which state.yaml already carries."""
    job, review, staging = run_job(tmp_path, OK_PY, share_logs=False)
    assert (review / "returncode.txt").exists()
    assert not (staging / "returncode.txt").exists()


def test_a_release_moves_the_logs_into_review(tmp_path):
    job, review, staging = run_job(tmp_path, OK_PY, share_logs=False)
    assert sorted(job.release_logs()) == ["stderr.txt", "stdout.txt"]
    assert (review / "stdout.txt").exists()
    assert not (staging / "stdout.txt").exists()


def test_the_default_releases_the_logs(tmp_path):
    """A caller that passes nothing keeps the behaviour it had before staging."""
    job, review, staging = run_job(tmp_path, OK_PY, share_logs=True)
    assert (review / "stdout.txt").exists()
    assert not (staging / "stdout.txt").exists()


def test_a_second_release_is_harmless(tmp_path):
    job, review, staging = run_job(tmp_path, OK_PY, share_logs=False)
    job.release_logs()
    assert job.release_logs() == []
    assert (review / "stdout.txt").exists()


def test_the_owner_reads_a_staged_log_through_the_viewer(tmp_path):
    job, review, staging = run_job(tmp_path, OK_PY, share_logs=False)
    assert "hello from job" in str(job.stdout)
    assert job.artifact_path("stdout.txt") == staging / "stdout.txt"
    job.release_logs()
    assert job.artifact_path("stdout.txt") == review / "stdout.txt"


def test_the_traceback_record_is_staged(tmp_path):
    job, review, staging = run_job(tmp_path, CRASH_PY, share_logs=False)
    record_path = staging / FRAMES_FILENAME
    assert record_path.exists()
    assert not (review / FRAMES_FILENAME).exists()

    record = json.loads(record_path.read_text())
    assert record["chain"][0]["type"] == "ValueError"
    assert "positive" not in record_path.read_text()

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


def test_rerun_clears_the_crash_record(tmp_path):
    """A crash record from the old run must not survive a later release.

    A rerun that succeeds writes no record, so a leftover record would describe
    a failure that the new run never had.
    """
    job, review, staging = run_job(tmp_path, CRASH_PY, share_logs=False)
    assert (staging / FRAMES_FILENAME).exists()

    job.rerun()

    assert not (staging / FRAMES_FILENAME).exists()
    assert not (review / FRAMES_FILENAME).exists()


def test_a_rerun_keeps_the_bundle_the_parties_approved(tmp_path):
    """The first run leaves .venv and any file the job wrote in code/.

    A rerun must judge frames against the tree as it stood before the first
    run, or a planted file counts as approved and the real sources drop out.
    """
    job, review, staging = run_job(tmp_path, OK_PY, share_logs=False)
    bundle_path = staging / BUNDLE_FILENAME
    assert bundle_path.exists()

    recorded = json.loads(bundle_path.read_text())
    assert "main.py" in recorded
    assert not any(name.startswith(".venv") for name in recorded)

    # The first run really did leave a virtual environment behind.
    code_dir = job.job_submission_path / "code"
    assert (code_dir / ".venv").is_dir()

    # A job that plants a file gains nothing on the next run.
    (code_dir / "DO2_row_4171_POSITIVE.py").write_text("\n" * 50)
    reloaded = load_or_create_bundle(bundle_path, code_dir)
    assert "DO2_row_4171_POSITIVE.py" not in reloaded
    assert reloaded == recorded
