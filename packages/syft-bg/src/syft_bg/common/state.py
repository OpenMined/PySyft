"""State management for tracking notified/approved entities."""

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError

from syft_bg.common.locking import file_lock


class BgState(BaseModel):
    """Typed top-level state for the background services.

    ``extra="allow"`` preserves the on-disk JSON format for arbitrary top-level
    keys written via :meth:`JsonStateManager.set_data` (e.g. ``snapshot``,
    ``peer_snapshot``, ``email_approve_last_history_id``).
    """

    model_config = ConfigDict(extra="allow")

    notified_jobs: dict[str, list[str]] = Field(default_factory=dict)
    approved_jobs: dict[str, dict[str, str]] = Field(default_factory=dict)
    approved_peers: dict[str, dict[str, str]] = Field(default_factory=dict)
    thread_ids: dict[str, str] = Field(default_factory=dict)


class JsonStateManager(BaseModel):
    """Manages state persistence with file locking for both notify and approve services."""

    state_file: Path
    _lock_file: Path = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self.state_file = self.state_file.expanduser()
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock_file = self.state_file.with_suffix(".lock")

        if not self.state_file.exists():
            self._save(BgState())

    @contextmanager
    def _file_lock(self):
        """Acquire exclusive file lock for thread-safe writes."""
        with file_lock(self._lock_file):
            yield

    def _load(self) -> BgState:
        if not self.state_file.exists():
            return BgState()
        try:
            return BgState.model_validate_json(self.state_file.read_text())
        except (ValidationError, json.JSONDecodeError, OSError):
            return BgState()

    def _save(self, state: BgState) -> None:
        with open(self.state_file, "w") as f:
            f.write(state.model_dump_json(indent=2))
        self.state_file.chmod(0o600)

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # --- Notification state (for syft-notify) ---

    def was_notified(self, entity_id: str, event_type: str) -> bool:
        """Check if entity was already notified for event type."""
        return event_type in self._load().notified_jobs.get(entity_id, [])

    def mark_notified(self, entity_id: str, event_type: str) -> None:
        """Mark entity as notified for event type."""
        with self._file_lock():
            state = self._load()
            events = state.notified_jobs.setdefault(entity_id, [])
            if event_type not in events:
                events.append(event_type)
            self._save(state)

    # --- Approval state (for syft-approve) ---

    def was_approved(self, job_name: str) -> bool:
        """Check if job was already approved."""
        return job_name in self._load().approved_jobs

    def mark_approved(self, job_name: str, submitted_by: str) -> None:
        """Mark job as approved."""
        with self._file_lock():
            state = self._load()
            state.approved_jobs[job_name] = {
                "approved_at": self._now_iso(),
                "submitted_by": submitted_by,
            }
            self._save(state)

    def get_approved_jobs(self) -> dict:
        """Get all approved jobs."""
        return self._load().approved_jobs

    def was_peer_approved(self, peer_email: str) -> bool:
        """Check if peer was already approved."""
        return f"peer_{peer_email}" in self._load().approved_peers

    def mark_peer_approved(self, peer_email: str, domain: str) -> None:
        """Mark peer as approved."""
        with self._file_lock():
            state = self._load()
            state.approved_peers[f"peer_{peer_email}"] = {
                "approved_at": self._now_iso(),
                "domain": domain,
            }
            self._save(state)

    def get_approved_peers(self) -> dict:
        """Get all approved peers."""
        return self._load().approved_peers

    # --- Email thread tracking ---

    def store_thread_id(self, job_name: str, thread_id: str) -> None:
        """Store Gmail thread ID for a job (for threaded notifications)."""
        with self._file_lock():
            state = self._load()
            state.thread_ids[job_name] = thread_id
            self._save(state)

    def get_thread_id(self, job_name: str) -> Optional[str]:
        """Get stored Gmail thread ID for a job."""
        return self._load().thread_ids.get(job_name)

    def get_job_name_by_thread_id(self, thread_id: str) -> Optional[str]:
        """Reverse lookup: find job_name for a given Gmail thread ID."""
        for job_name, tid in self._load().thread_ids.items():
            if tid == thread_id:
                return job_name
        return None

    # --- State inspection ---

    def is_empty(self) -> bool:
        """Check if state has no tracked entities (fresh state)."""
        state = self._load()
        # Consider empty if no notifications or approvals tracked
        has_notified = bool(state.notified_jobs)
        has_approved = bool(state.approved_jobs or state.approved_peers)
        return not (has_notified or has_approved)

    # --- Generic data storage ---

    def get_data(self, key: str, default: Optional[Any] = None) -> Any:
        """Get arbitrary data by key."""
        return self._load().model_dump(mode="json").get(key, default)

    def set_data(self, key: str, value: Any) -> None:
        """Set arbitrary data by key."""
        with self._file_lock():
            state = self._load()
            setattr(state, key, value)
            self._save(state)
