"""End-to-end unit test for the syft-job package lifecycle."""

import time
from pathlib import Path

from syft_job.client import JobClient
from syft_job.config import SyftJobConfig
from syft_job.job_runner import SyftJobRunner
from syft_perms import SyftPermContext
from syft_job.models.state import JobState


DO_EMAIL = "do@test.org"
DS_EMAIL = "ds@test.org"

MAIN_PY = """\
import os

print("hello from job")
os.makedirs("outputs", exist_ok=True)
with open("outputs/result.txt", "w") as f:
    f.write("done")
"""


def test_full_job_lifecycle(tmp_path: Path):
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()

    # Write a trivial main.py for the DS to submit
    code_file = tmp_path / "main.py"
    code_file.write_text(MAIN_PY)

    # Both configs share the same syftbox folder
    do_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DO_EMAIL)
    ds_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS_EMAIL)

    ds_client = JobClient(config=ds_config)
    do_client = JobClient(config=do_config)
    do_runner = SyftJobRunner(config=do_config)

    # --- DS submits a python job to DO ---
    job_dir = ds_client.submit_python_job(
        user=DO_EMAIL,
        code_path=str(code_file),
        job_name="test.job",
    )
    assert job_dir.exists()
    assert (job_dir / "run.sh").exists()
    assert (job_dir / "config.yaml").exists()
    assert (job_dir / "code" / "main.py").exists()

    # Job dir should be under inbox/<ds_email>/<job_name>
    expected_parent = ds_config.get_all_submissions_dir(DO_EMAIL) / DS_EMAIL
    assert job_dir.parent == expected_parent

    # --- DO lists jobs (auto-scans inbox) and sees it as pending ---
    review_path = do_config.get_review_job_dir(DO_EMAIL, DS_EMAIL, "test.job")
    jobs = do_client.jobs

    # Verify state.yaml was auto-created in review/
    assert (review_path / "state.yaml").exists()
    assert len(jobs) == 1
    job = jobs[0]
    assert job.name == "test.job"
    assert job.status == "pending"
    assert job.submitted_by == DS_EMAIL

    # --- DO approves the job ---
    job.approve()
    assert job.status == "approved"

    # --- DO runs approved jobs ---
    do_runner.process_approved_jobs(stream_output=False, timeout=120)

    # Re-fetch to get updated status
    job = do_client.jobs[0]
    assert job.status == "done"

    # --- Check outputs ---
    # output_paths includes syft.pub.yaml created by _prepare_outputs_dir
    output_names = {p.name for p in job.output_paths}
    assert "result.txt" in output_names
    result_file = next(p for p in job.output_paths if p.name == "result.txt")
    assert result_file.read_text().strip() == "done"

    # --- Check stdout / stderr (now in review/) ---
    stdout_path = review_path / "stdout.txt"
    stderr_path = review_path / "stderr.txt"
    assert stdout_path.exists()
    assert stderr_path.exists()
    assert "hello from job" in stdout_path.read_text()

    # --- Check returncode (now in review/) ---
    returncode_path = review_path / "returncode.txt"
    assert returncode_path.exists()
    assert returncode_path.read_text().strip() == "0"

    # --- Before sharing, DS should NOT have read access ---
    ctx = SyftPermContext(datasite=syftbox / DO_EMAIL)
    assert not ctx.open(
        f"app_data/job/review/{DS_EMAIL}/test.job/outputs/"
    ).has_read_access(DS_EMAIL)
    assert not ctx.open(
        f"app_data/job/review/{DS_EMAIL}/test.job/stdout.txt"
    ).has_read_access(DS_EMAIL)
    assert not ctx.open(
        f"app_data/job/review/{DS_EMAIL}/test.job/stderr.txt"
    ).has_read_access(DS_EMAIL)
    assert not ctx.open(
        f"app_data/job/review/{DS_EMAIL}/test.job/returncode.txt"
    ).has_read_access(DS_EMAIL)

    # --- Share outputs and logs with DS ---
    job.share_outputs([DS_EMAIL])
    job.share_logs([DS_EMAIL])

    # --- Verify DS has read access via SyftPermContext ---
    ctx = SyftPermContext(datasite=syftbox / DO_EMAIL)

    outputs_folder = ctx.open(f"app_data/job/review/{DS_EMAIL}/test.job/outputs/")
    assert outputs_folder.has_read_access(DS_EMAIL)

    stdout_file = ctx.open(f"app_data/job/review/{DS_EMAIL}/test.job/stdout.txt")
    assert stdout_file.has_read_access(DS_EMAIL)

    stderr_file = ctx.open(f"app_data/job/review/{DS_EMAIL}/test.job/stderr.txt")
    assert stderr_file.has_read_access(DS_EMAIL)

    returncode_file = ctx.open(
        f"app_data/job/review/{DS_EMAIL}/test.job/returncode.txt"
    )
    assert returncode_file.has_read_access(DS_EMAIL)


def test_ds_job_folder_permissions(tmp_path: Path):
    """Test that setup_ds_job_folder_as_do creates inbox and review folders with correct permissions."""
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()

    do_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DO_EMAIL)
    do_client = JobClient(config=do_config)

    # Create DS job folder with permissions
    ds_inbox_folder = do_client.setup_ds_job_folder_as_do(DS_EMAIL)
    assert ds_inbox_folder.exists()
    assert ds_inbox_folder == do_config.get_all_submissions_dir(DO_EMAIL) / DS_EMAIL

    # Review folder should also exist
    ds_review_folder = do_config.get_review_dir(DO_EMAIL) / DS_EMAIL
    assert ds_review_folder.exists()

    # DS should have write access to their inbox folder
    ctx = SyftPermContext(datasite=syftbox / DO_EMAIL)
    inbox_folder = ctx.open(f"app_data/job/inbox/{DS_EMAIL}/")
    assert inbox_folder.has_write_access(DS_EMAIL)
    assert inbox_folder.has_read_access(DO_EMAIL)

    # DS should have read access to their review folder
    review_folder = ctx.open(f"app_data/job/review/{DS_EMAIL}/")
    assert review_folder.has_read_access(DS_EMAIL)

    # Another user should NOT have write access
    other_email = "other@test.org"
    assert not inbox_folder.has_write_access(other_email)
    assert not inbox_folder.has_read_access(other_email)


def test_job_reject(tmp_path: Path):
    """Test that a DO can reject a pending job."""
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()

    code_file = tmp_path / "main.py"
    code_file.write_text("print('hello')")

    do_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DO_EMAIL)
    ds_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS_EMAIL)

    ds_client = JobClient(config=ds_config)
    do_client = JobClient(config=do_config)

    ds_client.submit_python_job(
        user=DO_EMAIL, code_path=str(code_file), job_name="reject.job"
    )
    job = do_client.jobs[0]  # auto-scans inbox
    assert job.status == "pending"

    job.reject(reason="Not approved")
    assert job.status == "rejected"
    assert job.review_reason == "Not approved"


def test_submission_validation(tmp_path: Path):
    """Test that invalid submissions are auto-rejected during scan_inbox."""
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()

    do_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DO_EMAIL)
    do_client = JobClient(config=do_config)

    # Manually create an invalid submission (missing code/ directory)
    submission_path = do_config.get_job_submission_dir(DO_EMAIL, DS_EMAIL, "bad.job")
    submission_path.mkdir(parents=True)
    (submission_path / "config.yaml").write_text(
        "name: bad.job\ntype: python\nsubmitted_by: ds@test.org\nsubmitted_at: '2025-01-01T00:00:00+00:00'\n"
    )
    (submission_path / "run.sh").write_text("#!/bin/bash\necho hi")
    # Missing code/ directory — should fail validation

    do_client.scan_inbox()

    review_state = (
        do_config.get_review_job_dir(DO_EMAIL, DS_EMAIL, "bad.job") / "state.yaml"
    )
    assert review_state.exists()

    state = JobState.load(review_state)
    assert state.status.value == "rejected"
    assert state.review_reason is not None


INFINITE_LOOP_PY = """\
while True:
    pass
"""


def test_timeout_does_not_hang_runner(tmp_path: Path):
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()

    code_file = tmp_path / "main.py"
    code_file.write_text(INFINITE_LOOP_PY)

    do_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DO_EMAIL)
    ds_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS_EMAIL)

    ds_client = JobClient(config=ds_config)
    do_client = JobClient(config=do_config)
    do_runner = SyftJobRunner(config=do_config)

    ds_client.submit_python_job(
        user=DO_EMAIL, code_path=str(code_file), job_name="loop.job"
    )
    do_client.jobs[0].approve()

    start = time.time()
    do_runner.process_approved_jobs(stream_output=True, timeout=3)
    elapsed = time.time() - start

    # 3s job timeout + venv setup + tree-kill cleanup should fit well under 60s.
    assert elapsed < 60, f"process_approved_jobs took {elapsed:.1f}s — likely hung"
    assert do_client.jobs[0].status == "failed"
