"""Activity feed widget for showing recent service activity."""

import re
from datetime import datetime

from textual.app import ComposeResult
from textual.containers import ScrollableContainer
from textual.widgets import Static

from syft_bg.services import ServiceManager


class ActivityFeed(Static):
    """Widget showing recent activity from all services."""

    DEFAULT_CSS = """
    ActivityFeed {
        width: 100%;
        height: 100%;
        background: $surface;
        border: solid $primary;
        padding: 1;
    }

    .activity-title {
        text-style: bold;
        margin-bottom: 1;
        color: $primary;
    }

    .activity-content {
        width: 100%;
        height: 100%;
        overflow-y: auto;
    }

    .activity-line {
        width: 100%;
    }

    .activity-line.notify {
        color: $primary;
    }

    .activity-line.approve {
        color: $success;
    }

    .activity-line.error {
        color: $error;
    }

    .activity-line.info {
        color: $text-muted;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.manager = ServiceManager()

    def compose(self) -> ComposeResult:
        yield Static("ACTIVITY FEED", classes="activity-title")
        yield ScrollableContainer(
            Static(self._get_activity_text(), id="activity-text"),
            classes="activity-content",
        )

    def _get_activity_text(self) -> str:
        """Get recent activity from all service logs."""
        all_entries = []

        for service_name in self.manager.list_services():
            logs = self.manager.get_logs(service_name, lines=20)
            for line in logs:
                # Parse timestamp and add service tag
                entry = self._parse_log_line(service_name, line)
                if entry:
                    all_entries.append(entry)

        # Sort by timestamp, most recent first
        all_entries.sort(key=lambda x: x[0], reverse=True)

        if not all_entries:
            return "No activity yet. Start services to see activity."

        # Format entries
        lines = []
        for timestamp, service, message in all_entries[:30]:
            time_str = timestamp.strftime("%H:%M:%S") if timestamp else "??:??:??"
            tag = self._get_service_tag(service)
            lines.append(f"{time_str} {tag} {message}")

        return "\n".join(lines)

    def _parse_log_line(self, service: str, line: str) -> tuple | None:
        """Parse a log line into (timestamp, service, message)."""
        if not line.strip():
            return None

        # Try to extract timestamp (common formats)
        timestamp = None
        message = line

        # Match ISO format: 2024-01-15 10:30:45
        iso_match = re.match(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", line)
        if iso_match:
            try:
                timestamp = datetime.strptime(iso_match.group(1), "%Y-%m-%d %H:%M:%S")
                message = line[iso_match.end() :].strip(" -:")
            except ValueError:
                pass

        # Match time only: 10:30:45
        time_match = re.match(r"(\d{2}:\d{2}:\d{2})", line)
        if not timestamp and time_match:
            try:
                today = datetime.now().date()
                time_part = datetime.strptime(time_match.group(1), "%H:%M:%S").time()
                timestamp = datetime.combine(today, time_part)
                message = line[time_match.end() :].strip(" -:")
            except ValueError:
                pass

        if not timestamp:
            timestamp = datetime.now()

        return (timestamp, service, message)

    def _get_service_tag(self, service: str) -> str:
        """Get a formatted tag for the service."""
        tags = {
            "notify": "[notify]",
            "approve": "[approve]",
        }
        return tags.get(service, f"[{service}]")

    def refresh_activity(self):
        """Refresh the activity feed."""
        try:
            text_widget = self.query_one("#activity-text", Static)
            text_widget.update(self._get_activity_text())
        except Exception:
            pass
