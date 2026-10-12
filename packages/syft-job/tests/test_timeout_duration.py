"""Tests for human-readable timeout durations in job runner messages."""

import sys
from pathlib import Path

from syft_job.config import SyftJobConfig
from syft_job.job_ref import JobRef
from syft_job.job_runner import SyftJobRunner, format_timeout_duration
from syft_job.migrations.registry import JOB_PROTOCOL_VERSION

DO_EMAIL = "do@test.org"
DS_EMAIL = "ds@test.org"

SLEEP_PY = """\
import time

print("working", flush=True)
time.sleep(30)
"""


def test_sub_minute_timeout_is_reported_in_seconds() -> None:
    assert format_timeout_duration(30) == "30 seconds"


def test_one_second_timeout_uses_singular() -> None:
    assert format_timeout_duration(1) == "1 second"


def test_boundary_below_one_minute_stays_in_seconds() -> None:
    assert format_timeout_duration(59) == "59 seconds"


def test_one_minute_timeout_uses_singular() -> None:
    assert format_timeout_duration(60) == "1 minute"


def test_sub_minute_remainder_truncates_like_before() -> None:
    # 61s previously printed "1 minutes" via 61 // 60; keep the truncation,
    # but with the correct plural.
    assert format_timeout_duration(61) == "1 minute"


def test_default_timeout_keeps_minutes_phrasing() -> None:
    assert format_timeout_duration(600) == "10 minutes"


def test_two_minute_timeout_is_plural() -> None:
    assert format_timeout_duration(120) == "2 minutes"


def test_captured_timeout_message_reports_seconds(tmp_path: Path, capsys) -> None:
    """A sub-minute timeout prints the duration in seconds, not "0 minutes"."""
    config = SyftJobConfig(syftbox_folder=tmp_path, current_user_email=DO_EMAIL)
    runner = SyftJobRunner(config=config)
    ref = JobRef(DO_EMAIL, DS_EMAIL, "short-timeout.job", JOB_PROTOCOL_VERSION)

    # Run main.py with the test's own Python, skipping the venv setup of a real job.
    run_dir = tmp_path / "run"
    staging_dir = runner.manager.staging_dir(ref)
    run_dir.mkdir()
    (run_dir / "main.py").write_text(SLEEP_PY)
    (run_dir / "run.sh").write_text(f'#!/bin/bash\n"{sys.executable}" main.py\n')

    returncode = runner._execute_job_captured(ref, timeout=1, run_dir=run_dir)

    assert returncode == -1
    stdout = (staging_dir / "stdout.txt").read_text()
    assert "--- PROCESS TIMED OUT ---" in stdout

    captured = capsys.readouterr().out
    assert "timed out after 1 second" in captured
    assert "0 minutes" not in captured
