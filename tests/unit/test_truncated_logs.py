"""Test for truncated logs when jobs have exceptions."""

import tempfile
from pathlib import Path
from syft_job.job_runner import SyftJobRunner
from syft_job.config import SyftJobConfig
from syft_job.models.state import JobState, JobStatus


def test_exception_logs_not_truncated():
    """Test that exception tracebacks are fully captured in stderr."""

    with tempfile.TemporaryDirectory() as tmpdir:
        syftbox_folder = Path(tmpdir) / "SyftBox"
        syftbox_folder.mkdir(parents=True)
        email = "test@example.com"
        ds_email = "ds@example.com"

        config = SyftJobConfig.from_syftbox_folder(str(syftbox_folder), email)

        job_name = "test_exception_job"

        # Create inbox dir with code/ and run.sh
        inbox_dir = config.get_job_submission_dir(email, ds_email, job_name)
        inbox_dir.mkdir(parents=True)
        code_dir = inbox_dir / "code"
        code_dir.mkdir()

        python_script = """
import sys
print("Starting job...", flush=True)
print("Line 2", flush=True)
print("Line 3", flush=True)

def func_a():
    func_b()

def func_b():
    func_c()

def func_c():
    func_d()

def func_d():
    raise ValueError("This is a test error with details: " + "x" * 50)

func_a()
"""
        (code_dir / "script.py").write_text(python_script)
        (inbox_dir / "run.sh").write_text("#!/bin/bash\npython code/script.py\n")
        (inbox_dir / "config.yaml").write_text(
            f"name: {job_name}\ntype: python\nsubmitted_by: {ds_email}\n"
        )

        # Create review dir with state.yaml (APPROVED)
        review_dir = config.get_review_job_dir(email, ds_email, job_name)
        review_dir.mkdir(parents=True)
        state = JobState(status=JobStatus.APPROVED)
        state.save(review_dir / "state.yaml")

        runner = SyftJobRunner(config)
        runner._execute_job(job_name, stream_output=True, timeout=30, user=ds_email)

        stdout_content = (review_dir / "stdout.txt").read_text()
        stderr_content = (review_dir / "stderr.txt").read_text()

        # Check stdout has all print statements
        assert "Starting job..." in stdout_content
        assert "Line 2" in stdout_content
        assert "Line 3" in stdout_content

        # Check stderr has full traceback
        assert "Traceback (most recent call last)" in stderr_content
        assert "func_a" in stderr_content
        assert "func_d" in stderr_content
        assert "ValueError" in stderr_content
        assert "This is a test error" in stderr_content
