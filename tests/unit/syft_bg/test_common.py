"""Tests for common module."""

import json
from pathlib import Path


from syft_bg.common.config import DefaultPaths, get_default_paths
from syft_bg.common.state import JsonStateManager


class TestDefaultPaths:
    """Tests for DefaultPaths dataclass."""

    def test_get_default_paths_returns_dataclass(self):
        """get_default_paths should return a DefaultPaths instance."""
        paths = get_default_paths()
        assert isinstance(paths, DefaultPaths)

    def test_paths_are_path_objects(self):
        """All paths should be Path objects."""
        paths = get_default_paths()
        assert isinstance(paths.config, Path)
        assert isinstance(paths.notify_state, Path)
        assert isinstance(paths.approve_state, Path)
        assert isinstance(paths.notify_pid, Path)
        assert isinstance(paths.approve_pid, Path)

    def test_paths_have_expected_structure(self):
        """Paths should follow expected naming conventions."""
        paths = get_default_paths()
        assert paths.config.name == "config.yaml"
        assert paths.notify_state.name == "state.json"
        assert paths.approve_state.name == "state.json"
        assert paths.notify_pid.name == "daemon.pid"
        assert paths.approve_pid.name == "daemon.pid"

    def test_notify_and_approve_have_separate_dirs(self):
        """Notify and approve should use separate directories."""
        paths = get_default_paths()
        assert paths.notify_state.parent != paths.approve_state.parent
        assert "notify" in str(paths.notify_state)
        assert "approve" in str(paths.approve_state)


class TestJsonStateManager:
    """Tests for JsonStateManager."""

    def test_empty_state_file(self, temp_dir):
        """Should handle non-existent state file."""
        state_path = temp_dir / "state.json"
        state = JsonStateManager(state_path)

        assert not state.was_notified("job1", "new_job")
        assert not state.was_approved("job1")

    def test_mark_and_check_notified(self, temp_dir):
        """Should track notified jobs by event type."""
        state_path = temp_dir / "state.json"
        state = JsonStateManager(state_path)

        assert not state.was_notified("job1", "new_job")
        state.mark_notified("job1", "new_job")
        assert state.was_notified("job1", "new_job")
        # Different event type should not be marked
        assert not state.was_notified("job1", "approved")

    def test_mark_and_check_approved(self, temp_dir):
        """Should track approved jobs."""
        state_path = temp_dir / "state.json"
        state = JsonStateManager(state_path)

        assert not state.was_approved("job1")
        state.mark_approved("job1", "user@example.com")
        assert state.was_approved("job1")

    def test_state_persists_to_file(self, temp_dir):
        """State should persist to disk."""
        state_path = temp_dir / "state.json"

        # Write state
        state1 = JsonStateManager(state_path)
        state1.mark_notified("job1", "new_job")

        # Read with new instance
        state2 = JsonStateManager(state_path)
        assert state2.was_notified("job1", "new_job")

    def test_state_file_is_valid_json(self, temp_dir):
        """State file should be valid JSON."""
        state_path = temp_dir / "state.json"
        state = JsonStateManager(state_path)
        state.mark_notified("job1", "new_job")

        # Should be parseable as JSON
        data = json.loads(state_path.read_text())
        assert "notified_jobs" in data
        assert "job1" in data["notified_jobs"]
        assert "new_job" in data["notified_jobs"]["job1"]

    def test_get_and_set_data(self, temp_dir):
        """Should support generic get/set operations."""
        state_path = temp_dir / "state.json"
        state = JsonStateManager(state_path)

        assert state.get_data("custom_key") is None
        state.set_data("custom_key", {"foo": "bar"})
        assert state.get_data("custom_key") == {"foo": "bar"}

    def test_store_and_get_thread_id(self, temp_dir):
        """Should store a thread_id for a job and retrieve it."""
        state_path = temp_dir / "state.json"
        state = JsonStateManager(state_path)

        state.store_thread_id("job1", "thread-abc-123")
        assert state.get_thread_id("job1") == "thread-abc-123"

    def test_get_thread_id_nonexistent(self, temp_dir):
        """Should return None for an unknown job's thread_id."""
        state_path = temp_dir / "state.json"
        state = JsonStateManager(state_path)

        assert state.get_thread_id("nonexistent_job") is None
