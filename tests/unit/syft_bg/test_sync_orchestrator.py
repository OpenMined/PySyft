"""Tests for SyncOrchestrator."""

from unittest.mock import MagicMock

from syft_bg.common.state import JsonStateManager
from syft_bg.sync.config import SyncConfig
from syft_bg.sync.orchestrator import SyncOrchestrator
from syft_bg.sync.snapshot import SyncSnapshot


def _make_orchestrator(temp_dir, **config_overrides):
    syft_client = MagicMock()
    syft_client.job_client.jobs = []
    syft_client.peer_manager.approved_peers = []
    syft_client.peer_manager.requested_by_peer_peers = []

    state = JsonStateManager(temp_dir / "sync_state.json")
    defaults = {"interval": 1, "max_retries": 2, "retry_backoff": 0.01}
    defaults.update(config_overrides)
    config = SyncConfig(**defaults)

    orch = SyncOrchestrator(syft_client, state, config)
    return orch, syft_client, state


def _read_snapshot(state: JsonStateManager) -> SyncSnapshot | None:
    data = state.get_data("snapshot")
    if not data:
        return None
    return SyncSnapshot.model_validate(data)


class TestRunOnce:
    def test_syncs_and_writes_snapshot(self, temp_dir):
        orch, client, state = _make_orchestrator(temp_dir)
        orch.run_once()
        client.sync.assert_called_once()
        snap = _read_snapshot(state)
        assert snap is not None
        assert snap.sync_count == 1

    def test_captures_job_names(self, temp_dir):
        orch, client, state = _make_orchestrator(temp_dir)
        mock_job = MagicMock()
        mock_job.name = "test_job"
        client.job_client.jobs = [mock_job]
        orch.run_once()
        snap = _read_snapshot(state)
        assert "test_job" in snap.job_names

    def test_captures_approved_peers(self, temp_dir):
        orch, client, state = _make_orchestrator(temp_dir)
        mock_peer = MagicMock()
        mock_peer.email = "ds@test.com"
        client.peer_manager.approved_peers = [mock_peer]
        orch.run_once()
        snap = _read_snapshot(state)
        assert "ds@test.com" in snap.approved_peer_emails

    def test_captures_all_peers(self, temp_dir):
        orch, client, state = _make_orchestrator(temp_dir)
        approved = MagicMock()
        approved.email = "ds@test.com"
        pending = MagicMock()
        pending.email = "new@test.com"
        client.peer_manager.approved_peers = [approved]
        client.peer_manager.requested_by_peer_peers = [pending]
        orch.run_once()
        snap = _read_snapshot(state)
        assert "ds@test.com" in snap.peer_emails
        assert "new@test.com" in snap.peer_emails
        assert "ds@test.com" in snap.approved_peer_emails
        assert "new@test.com" not in snap.approved_peer_emails


class TestSyncFailure:
    def test_records_error_in_snapshot(self, temp_dir):
        orch, client, state = _make_orchestrator(temp_dir)
        client.sync.side_effect = Exception("Drive down")
        orch.run_once()
        snap = _read_snapshot(state)
        assert snap is not None
        assert "Drive down" in snap.sync_error

    def test_retries_before_failing(self, temp_dir):
        orch, client, state = _make_orchestrator(temp_dir)
        client.sync.side_effect = [Exception("fail"), None]
        orch.run_once()
        assert client.sync.call_count == 2
        snap = _read_snapshot(state)
        assert snap.sync_error is None

    def test_exhausts_retries(self, temp_dir):
        orch, client, state = _make_orchestrator(temp_dir)
        client.sync.side_effect = Exception("always fails")
        orch.run_once()
        assert client.sync.call_count == 2  # max_retries=2
        snap = _read_snapshot(state)
        assert snap.sync_error is not None
