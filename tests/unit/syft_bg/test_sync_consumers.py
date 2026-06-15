"""Tests for consumer services wired to sync state."""

import time
from unittest.mock import MagicMock

from syft_bg.common.state import JsonStateManager
from syft_bg.notify.monitors.job import JobMonitor
from syft_bg.notify.monitors.peer import PeerMonitor
from syft_bg.sync.snapshot import SyncSnapshot


def _write_snapshot(state: JsonStateManager, **kwargs) -> None:
    defaults = {"sync_time": time.time(), "sync_count": 1}
    defaults.update(kwargs)
    snapshot = SyncSnapshot(**defaults)
    state.set_data("snapshot", snapshot.model_dump())


class TestPeerMonitorWithSnapshot:
    def test_reads_peers_from_snapshot(self, temp_dir):
        sync_state = JsonStateManager(temp_dir / "sync_state.json")
        _write_snapshot(sync_state, peer_emails=["ds@test.com"])
        handler = MagicMock()
        handler.on_new_peer_request_to_do.return_value = True
        handler.on_peer_request_sent.return_value = True
        state = JsonStateManager(temp_dir / "state.json")

        monitor = PeerMonitor(
            do_email="do@test.com",
            handler=handler,
            state=state,
            sync_state=sync_state,
        )
        monitor._check_all_entities()
        handler.on_new_peer_request_to_do.assert_called_once_with(
            "do@test.com", "ds@test.com"
        )

    def test_reads_approved_peers_from_snapshot(self, temp_dir):
        sync_state = JsonStateManager(temp_dir / "sync_state.json")
        _write_snapshot(sync_state, approved_peer_emails=["ds@test.com"])
        handler = MagicMock()
        handler.on_peer_granted.return_value = True
        state = JsonStateManager(temp_dir / "state.json")

        monitor = PeerMonitor(
            do_email="do@test.com",
            handler=handler,
            state=state,
            sync_state=sync_state,
        )
        monitor._check_all_entities()
        handler.on_peer_granted.assert_called_once_with("ds@test.com", "do@test.com")

    def test_missing_snapshot_returns_empty_peers(self, temp_dir):
        sync_state = JsonStateManager(temp_dir / "sync_state.json")
        handler = MagicMock()
        state = JsonStateManager(temp_dir / "state.json")

        monitor = PeerMonitor(
            do_email="do@test.com",
            handler=handler,
            state=state,
            sync_state=sync_state,
        )
        monitor._check_all_entities()
        handler.on_new_peer_request_to_do.assert_not_called()
        handler.on_peer_granted.assert_not_called()


class TestJobMonitorLocalStatusChanges:
    """Tests for process_local_status_changes with inbox/review directory structure."""

    def _setup_job(self, temp_dir, ds_email="ds@t.com", job_name="test_job"):
        do_email = "do@test.com"
        job_dir = temp_dir / do_email / "app_data" / "job"

        inbox_job = job_dir / "inbox" / ds_email / job_name
        inbox_job.mkdir(parents=True)
        config = inbox_job / "config.yaml"
        config.write_text(
            f"name: {job_name}\ntype: python\nsubmitted_at: '2026-01-01T00:00:00'\n"
        )

        return job_dir, inbox_job

    def test_detects_new_job(self, temp_dir):
        self._setup_job(temp_dir)
        handler = MagicMock()
        handler.on_new_job.return_value = True
        state = JsonStateManager(temp_dir / "state.json")
        state.mark_notified("_dummy", "x")  # non-empty state -> not fresh

        monitor = JobMonitor(
            syftbox_root=temp_dir,
            do_email="do@test.com",
            handler=handler,
            state=state,
        )
        monitor.process_local_status_changes()
        handler.on_new_job.assert_called_once_with(
            "do@test.com", "test_job", "ds@t.com"
        )

    def _write_review_state(self, temp_dir, ds_email, job_name, status):
        review_dir = (
            temp_dir
            / "do@test.com"
            / "app_data"
            / "job"
            / "review"
            / ds_email
            / job_name
        )
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / "state.yaml").write_text(f"status: {status}\n")

    def test_detects_approved_status(self, temp_dir):
        self._setup_job(temp_dir)
        self._write_review_state(temp_dir, "ds@t.com", "test_job", "approved")
        handler = MagicMock()
        handler.on_new_job.return_value = True
        handler.on_job_approved.return_value = True
        state = JsonStateManager(temp_dir / "state.json")
        state.mark_notified("_dummy", "x")

        monitor = JobMonitor(
            syftbox_root=temp_dir,
            do_email="do@test.com",
            handler=handler,
            state=state,
        )
        monitor.process_local_status_changes()
        handler.on_job_approved.assert_called_once_with("ds@t.com", "test_job")

    def test_detects_executed_status(self, temp_dir):
        self._setup_job(temp_dir)
        self._write_review_state(temp_dir, "ds@t.com", "test_job", "done")
        handler = MagicMock()
        handler.on_new_job.return_value = True
        handler.on_job_approved.return_value = True
        handler.on_job_executed.return_value = True
        state = JsonStateManager(temp_dir / "state.json")
        state.mark_notified("_dummy", "x")

        monitor = JobMonitor(
            syftbox_root=temp_dir,
            do_email="do@test.com",
            handler=handler,
            state=state,
        )
        monitor.process_local_status_changes()
        handler.on_job_approved.assert_called_once_with("ds@t.com", "test_job")
        handler.on_job_executed.assert_called_once_with("ds@t.com", "test_job")

    def test_skips_old_jobs_on_fresh_state(self, temp_dir):
        self._setup_job(temp_dir)
        handler = MagicMock()
        state = JsonStateManager(temp_dir / "state.json")  # empty = fresh

        monitor = JobMonitor(
            syftbox_root=temp_dir,
            do_email="do@test.com",
            handler=handler,
            state=state,
        )
        monitor.seed_existing_jobs()
        monitor.process_local_status_changes()
        handler.on_new_job.assert_not_called()

    def test_no_inbox_dir_doesnt_crash(self, temp_dir):
        handler = MagicMock()
        state = JsonStateManager(temp_dir / "state.json")

        monitor = JobMonitor(
            syftbox_root=temp_dir,
            do_email="do@test.com",
            handler=handler,
            state=state,
        )
        monitor.process_local_status_changes()  # should not raise
        handler.on_new_job.assert_not_called()
