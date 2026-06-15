from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel


class JobStatus(str, Enum):
    """Status of a job in the system."""

    RECEIVED = "received"  # files arrived, not yet validated
    PENDING = "pending"  # validated, awaiting DO review
    APPROVED = "approved"  # DO approved, ready for execution
    REJECTED = "rejected"  # DO rejected
    RUNNING = "running"  # runner is executing
    DONE = "done"  # completed successfully
    FAILED = "failed"  # execution failed


class JobState(BaseModel):
    """Represents the state of a job, stored as state.yaml in the review/ directory."""

    status: JobStatus = JobStatus.RECEIVED
    received_at: Optional[datetime] = None

    # Local job approval
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    approval_method: Optional[str] = None  # "manual" or "auto"

    # Rejection
    rejected_by: Optional[str] = None
    rejected_at: Optional[datetime] = None

    # Reason supplied at review time (approve or reject)
    review_reason: Optional[str] = None

    # Completion
    completed_at: Optional[datetime] = None
    return_code: Optional[int] = None

    def save(self, path: Path) -> None:
        """Write state to a YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(mode="json"), f, default_flow_style=False)

    @classmethod
    def load(cls, path: Path) -> JobState:
        """Load state from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
