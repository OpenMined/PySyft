"""Tests for JobHandler notification handler."""

from unittest.mock import MagicMock

import pytest

from syft_bg.notify.gmail.sender import SendResult
from syft_bg.notify.handlers.job import (
    _MAX_STDERR_SIZE,
    JobHandler,
    _friendly_reason,
    _read_job_stderr,
)
from syft_job.job import JobInfo


class TestJobHandler:
    """Tests for the JobHandler class."""

    def _make_handler(self, **kwargs):
        """Create a JobHandler with mocked dependencies."""
        sender = MagicMock()
        state = MagicMock()
        state.was_notified.return_value = False
        state.get_thread_id.return_value = "thread-123"

        defaults = dict(
            sender=sender,
            state=state,
            do_email="do@test.com",
            notify_on_new=True,
            notify_on_approved=True,
            notify_on_executed=True,
        )
        defaults.update(kwargs)

        handler = JobHandler(**defaults)
        return handler, sender, state

    def test_on_new_job(self):
        """on_new_job should call sender.notify_new_job and store thread_id."""
        handler, sender, state = self._make_handler()

        sender.notify_new_job.return_value = SendResult(
            success=True, thread_id="thread-123"
        )

        result = handler.on_new_job("do@test.com", "job1", "ds@test.com")

        assert result is True
        sender.notify_new_job.assert_called_once_with(
            "do@test.com",
            "job1",
            "ds@test.com",
            job_url=None,
            job_code=None,
        )
        state.mark_notified.assert_called_once_with("job1", "new")
        state.store_thread_id.assert_called_once_with("job1", "thread-123")

    def test_on_job_approved(self):
        """on_job_approved should notify DS and also notify DO in same thread."""
        handler, sender, state = self._make_handler()

        sender.notify_job_approved.return_value = SendResult(
            success=True, thread_id="thread-456"
        )

        result = handler.on_job_approved("ds@test.com", "job1")

        assert result is True
        sender.notify_job_approved.assert_called_once_with(
            "ds@test.com", "job1", job_url=None
        )
        state.mark_notified.assert_called_once_with("job1", "approved")
        # Should also notify DO in the same thread
        sender.notify_job_approved_to_do.assert_called_once_with(
            "do@test.com", "job1", "ds@test.com", thread_id="thread-123"
        )

    def test_on_job_executed(self):
        """on_job_executed should notify DS and also notify DO in same thread."""
        handler, sender, state = self._make_handler()

        sender.notify_job_executed.return_value = SendResult(
            success=True, thread_id="thread-789"
        )

        result = handler.on_job_executed("ds@test.com", "job1", duration=42)

        assert result is True
        sender.notify_job_executed.assert_called_once_with(
            "ds@test.com", "job1", duration=42, results_url=None
        )
        state.mark_notified.assert_called_once_with("job1", "executed")
        # Should also notify DO in the same thread
        sender.notify_job_completed_to_do.assert_called_once_with(
            "do@test.com",
            "job1",
            "ds@test.com",
            duration=42,
            thread_id="thread-123",
        )

    def test_on_job_rejected_notifies_do_and_ds(self):
        """on_job_rejected should notify DO and DS."""
        handler, sender, state = self._make_handler()

        sender.notify_job_rejected_to_do.return_value = SendResult(
            success=True, thread_id="thread-rej"
        )

        result = handler.on_job_rejected(
            "do@test.com", "job1", "ds@test.com", "hash mismatch for main.py"
        )

        assert result is True
        sender.notify_job_rejected_to_do.assert_called_once_with(
            "do@test.com",
            "job1",
            "ds@test.com",
            "hash mismatch for main.py",
            thread_id="thread-123",
        )
        state.mark_notified.assert_called_once_with("job1", "rejected")
        # DS should also be notified with a friendly message
        sender.notify_job_rejected_to_ds.assert_called_once()
        ds_call = sender.notify_job_rejected_to_ds.call_args
        assert ds_call[0][0] == "ds@test.com"
        assert ds_call[0][1] == "job1"
        assert "differs from the approved version" in ds_call[0][2]

    def test_on_job_rejected_ds_not_notified_on_failure(self):
        """DS should not be notified if DO notification fails."""
        handler, sender, state = self._make_handler()

        sender.notify_job_rejected_to_do.return_value = SendResult(success=False)

        result = handler.on_job_rejected(
            "do@test.com", "job1", "ds@test.com", "unknown peer: ds@test.com"
        )

        assert result is False
        sender.notify_job_rejected_to_ds.assert_not_called()


class TestFriendlyReason:
    """Tests for _friendly_reason helper."""

    def test_unknown_peer(self):
        msg = _friendly_reason("unknown peer: ds@test.com", "job1")
        assert "not yet registered" in msg
        assert "job1" in msg

    def test_unapproved_file(self):
        msg = _friendly_reason("unapproved file: extra.py", "job1")
        assert "hasn't been approved" in msg

    def test_hash_mismatch(self):
        msg = _friendly_reason(
            "script hash mismatch for main.py: expected sha256:aaa, got sha256:bbb",
            "job1",
        )
        assert "differs from the approved version" in msg

    def test_no_python_files(self):
        msg = _friendly_reason("no Python files found in submission", "job1")
        assert "did not contain any Python files" in msg

    def test_generic_fallback(self):
        msg = _friendly_reason("some other reason", "job1")
        assert "was not approved" in msg
        assert "some other reason" in msg


def _client_for_staged_job(tmp_path):
    """A job client with one job. Returns (job_client, staging_dir)."""
    review = tmp_path / "review"
    staging = tmp_path / "staging"
    review.mkdir()
    staging.mkdir()

    job = MagicMock()
    job.name = "job1"
    job.job_review_path = review
    job.job_staging_path = staging
    job.artifact_path = lambda name: JobInfo.artifact_path(job, name)
    job_client = MagicMock()
    job_client.jobs = [job]
    return job_client, staging


def test_read_job_stderr_finds_staged_artifacts(tmp_path):
    """A failed job keeps stderr and returncode.txt in staging until a release."""
    job_client, staging = _client_for_staged_job(tmp_path)
    (staging / "stderr.txt").write_text("boom")
    (staging / "returncode.txt").write_text("3")

    assert _read_job_stderr(job_client, "job1") == ("boom", 3)


@pytest.mark.parametrize(
    "make_stderr",
    [
        lambda path: None,
        lambda path: path.write_text("  \n"),
        lambda path: path.mkdir(),
    ],
    ids=["missing", "empty", "unreadable"],
)
def test_read_job_stderr_without_usable_output(tmp_path, make_stderr):
    job_client, staging = _client_for_staged_job(tmp_path)
    make_stderr(staging / "stderr.txt")

    assert _read_job_stderr(job_client, "job1") == (None, None)


def test_read_job_stderr_truncates_to_tail(tmp_path):
    job_client, staging = _client_for_staged_job(tmp_path)
    head = "h" * _MAX_STDERR_SIZE
    tail = "t" * (_MAX_STDERR_SIZE - 3) + "END"
    (staging / "stderr.txt").write_text(head + tail)

    stderr_text, _ = _read_job_stderr(job_client, "job1")

    total_kb = 2 * _MAX_STDERR_SIZE // 1024
    assert stderr_text == (
        f"[... truncated, showing last {_MAX_STDERR_SIZE // 1024} KB "
        f"of {total_kb} KB total]\n\n{tail}"
    )
