"""State management for tracking notified/approved entities."""

import fcntl
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class JsonStateManager:
    """Manages state persistence with file locking for both notify and approve services."""

    def __init__(self, state_file: Path):
        self.state_file = Path(state_file).expanduser()
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock_file = self.state_file.with_suffix(".lock")

        if not self.state_file.exists():
            self._save_all({})

    @contextmanager
    def _file_lock(self):
        """Acquire exclusive file lock for thread-safe writes."""
        self._lock_file.touch(exist_ok=True)
        with open(self._lock_file, "r") as lock_handle:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _load_all(self) -> dict:
        if not self.state_file.exists():
            return {}
        try:
            with open(self.state_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError, IOError):
            return {}

    def _save_all(self, data: dict):
        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=2)
        self.state_file.chmod(0o600)

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # --- Notification state (for syft-notify) ---

    def was_notified(self, entity_id: str, event_type: str) -> bool:
        """Check if entity was already notified for event type."""
        data = self._load_all()
        notified = data.get("notified_jobs", {})
        return event_type in notified.get(entity_id, [])

    def mark_notified(self, entity_id: str, event_type: str) -> None:
        """Mark entity as notified for event type."""
        with self._file_lock():
            data = self._load_all()
            if "notified_jobs" not in data:
                data["notified_jobs"] = {}
            if entity_id not in data["notified_jobs"]:
                data["notified_jobs"][entity_id] = []
            if event_type not in data["notified_jobs"][entity_id]:
                data["notified_jobs"][entity_id].append(event_type)
            self._save_all(data)

    # --- Approval state (for syft-approve) ---

    def was_approved(self, job_name: str) -> bool:
        """Check if job was already approved."""
        data = self._load_all()
        return job_name in data.get("approved_jobs", {})

    def mark_approved(self, job_name: str, submitted_by: str) -> None:
        """Mark job as approved."""
        with self._file_lock():
            data = self._load_all()
            if "approved_jobs" not in data:
                data["approved_jobs"] = {}
            data["approved_jobs"][job_name] = {
                "approved_at": self._now_iso(),
                "submitted_by": submitted_by,
            }
            self._save_all(data)

    def get_approved_jobs(self) -> dict:
        """Get all approved jobs."""
        return self._load_all().get("approved_jobs", {})

    def was_peer_approved(self, peer_email: str) -> bool:
        """Check if peer was already approved."""
        data = self._load_all()
        return f"peer_{peer_email}" in data.get("approved_peers", {})

    def mark_peer_approved(self, peer_email: str, domain: str) -> None:
        """Mark peer as approved."""
        with self._file_lock():
            data = self._load_all()
            if "approved_peers" not in data:
                data["approved_peers"] = {}
            data["approved_peers"][f"peer_{peer_email}"] = {
                "approved_at": self._now_iso(),
                "domain": domain,
            }
            self._save_all(data)

    def get_approved_peers(self) -> dict:
        """Get all approved peers."""
        return self._load_all().get("approved_peers", {})

    # --- Email thread tracking ---

    def store_thread_id(self, job_name: str, thread_id: str) -> None:
        """Store Gmail thread ID for a job (for threaded notifications)."""
        with self._file_lock():
            data = self._load_all()
            if "thread_ids" not in data:
                data["thread_ids"] = {}
            data["thread_ids"][job_name] = thread_id
            self._save_all(data)

    def get_thread_id(self, job_name: str) -> Optional[str]:
        """Get stored Gmail thread ID for a job."""
        data = self._load_all()
        return data.get("thread_ids", {}).get(job_name)

    def get_job_name_by_thread_id(self, thread_id: str) -> Optional[str]:
        """Reverse lookup: find job_name for a given Gmail thread ID."""
        data = self._load_all()
        for job_name, tid in data.get("thread_ids", {}).items():
            if tid == thread_id:
                return job_name
        return None

    # --- State inspection ---

    def is_empty(self) -> bool:
        """Check if state has no tracked entities (fresh state)."""
        data = self._load_all()
        # Consider empty if no notifications or approvals tracked
        has_notified = bool(data.get("notified_jobs"))
        has_approved = bool(data.get("approved_jobs") or data.get("approved_peers"))
        return not (has_notified or has_approved)

    # --- Generic data storage ---

    def get_data(self, key: str, default: Optional[Any] = None) -> Any:
        """Get arbitrary data by key."""
        return self._load_all().get(key, default)

    def set_data(self, key: str, value: Any) -> None:
        """Set arbitrary data by key."""
        with self._file_lock():
            data = self._load_all()
            data[key] = value
            self._save_all(data)
