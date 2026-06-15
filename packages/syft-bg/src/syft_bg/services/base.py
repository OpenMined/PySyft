"""Base service definition for background services."""

import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class ServiceStatus(str, Enum):
    STARTING = "starting"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    UNKNOWN = "unknown"


class ServiceInfo(BaseModel):
    name: str
    status: ServiceStatus
    pid: Optional[int] = None
    error: Optional[str] = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    installed: bool = False

    @property
    def status_str(self) -> str:
        if self.status == ServiceStatus.STARTING:
            return f"starting (PID {self.pid})"
        if self.status == ServiceStatus.RUNNING:
            return f"running (PID {self.pid})"
        if self.status == ServiceStatus.ERROR:
            return "setup error"
        return "stopped"

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(exclude={"pid", "installed"}, indent=2))

    @classmethod
    def load(cls, path: Path) -> Optional["ServiceInfo"]:
        if not path.exists():
            return None
        try:
            return cls.model_validate_json(path.read_text())
        except Exception:
            return None


class Service:
    """A background service that runs as a subprocess."""

    def __init__(
        self,
        name: str,
        description: str,
        pid_file: Path,
        log_file: Path,
    ):
        self.name = name
        self.description = description
        self.pid_file = pid_file
        self.log_file = log_file

    def get_pid(self) -> Optional[int]:
        """Get the PID from the pid file."""
        if not self.pid_file.exists():
            return None
        try:
            return int(self.pid_file.read_text().strip())
        except (ValueError, OSError):
            return None

    def is_running(self) -> bool:
        """Check if the service is running."""
        pid = self.get_pid()
        if not pid:
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def info(self) -> ServiceInfo:
        """Get the current service status."""
        from syft_bg.api.utils import load_setup_state
        from syft_bg.systemd import is_installed

        persisted = load_setup_state(self.name)
        pid = self.get_pid()
        process_running = bool(pid and self.is_running())

        if persisted and persisted.status == ServiceStatus.ERROR:
            status = ServiceStatus.ERROR
        elif not process_running:
            status = ServiceStatus.STOPPED
        elif persisted and persisted.status == ServiceStatus.STARTING:
            status = ServiceStatus.STARTING
        else:
            status = ServiceStatus.RUNNING

        return ServiceInfo(
            name=self.name,
            status=status,
            pid=pid if process_running else None,
            error=persisted.error if persisted else None,
            timestamp=(
                persisted.timestamp
                if persisted
                else datetime.now(timezone.utc).isoformat()
            ),
            installed=is_installed(self.name),
        )

    def start(self) -> tuple[bool, str]:
        """Start the service as a background subprocess."""
        if self.is_running():
            print(f"{self.name} is already running")

        # Ensure directories exist
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Open log file for output
        log_fd = open(self.log_file, "a")

        # Spawn a subprocess that runs the orchestrator via the API
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["SYFT_BG_DAEMON"] = "1"
        script = (
            f"from syft_bg.api.api import run_foreground; run_foreground('{self.name}')"
        )
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", script],
            stdout=log_fd,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )

        # Write PID file
        self.pid_file.write_text(str(process.pid))

    def stop(self) -> tuple[bool, str]:
        """Stop the service."""
        if not self.is_running():
            # Clean up stale PID file
            if self.pid_file.exists():
                self.pid_file.unlink()
            raise ValueError(f"{self.name} is not running")

        pid = self.get_pid()
        if not pid:
            return (False, "Could not get PID")

        # Send SIGTERM for graceful shutdown
        os.kill(pid, signal.SIGTERM)

        # Wait briefly for process to terminate
        import time

        for _ in range(10):  # Wait up to 1 second
            time.sleep(0.1)
            try:
                os.kill(pid, 0)
            except OSError:
                break  # Process terminated
        else:
            # Force kill if still running
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass

        # Clean up PID file
        if self.pid_file.exists():
            self.pid_file.unlink()

    def restart(self) -> tuple[bool, str]:
        """Restart the service."""
        if self.is_running():
            self.stop()

        self.start()

    def get_logs(self, lines: int = 50) -> list[str]:
        """Get recent log lines."""
        if not self.log_file.exists():
            return []
        try:
            all_lines = self.log_file.read_text().strip().split("\n")
            return all_lines[-lines:]
        except Exception:
            return []
