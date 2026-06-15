"""Service card widget for displaying service status."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Static
from syft_bg.services.base import ServiceStatus
from syft_bg.services import ServiceManager


class ServiceCard(Static):
    """Widget displaying a single service's status."""

    DEFAULT_CSS = """
    ServiceCard {
        width: 100%;
        height: auto;
        padding: 1 2;
        margin: 0 0 1 0;
        background: $surface;
        border: solid $primary;
    }

    ServiceCard.running {
        border: solid $success;
    }

    ServiceCard.stopped {
        border: solid $surface-lighten-2;
    }

    ServiceCard.error {
        border: solid $error;
    }

    ServiceCard:focus {
        border: double $accent;
    }

    .service-header {
        width: 100%;
    }

    .service-name {
        text-style: bold;
        width: auto;
    }

    .service-status {
        width: auto;
        margin-left: 2;
    }

    .service-status.running {
        color: $success;
    }

    .service-status.stopped {
        color: $text-muted;
    }

    .service-status.error {
        color: $error;
    }

    .service-desc {
        color: $text-muted;
        margin-top: 1;
    }

    .service-pid {
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("enter", "toggle", "Start/Stop"),
        Binding("r", "restart", "Restart"),
        Binding("l", "logs", "Logs"),
    ]

    can_focus = True

    def __init__(self, service_name: str, **kwargs):
        super().__init__(**kwargs)
        self.service_name = service_name
        self.manager = ServiceManager()

    def compose(self) -> ComposeResult:
        service = self.manager.get_service(self.service_name)
        info = service.info()

        status_text = self._get_status_text(info.status)
        status_class = self._get_status_class(info.status)
        pid_text = f"PID: {info.pid}" if info.pid else ""

        with Horizontal(classes="service-header"):
            yield Static(self.service_name.upper(), classes="service-name")
            yield Static(
                f"● {status_text}"
                if info.status == ServiceStatus.RUNNING
                else f"○ {status_text}",
                classes=f"service-status {status_class}",
                id=f"status-{self.service_name}",
            )
            yield Static(pid_text, classes="service-pid", id=f"pid-{self.service_name}")

        yield Static(service.description, classes="service-desc")

    def _get_status_text(self, status: ServiceStatus) -> str:
        return {
            ServiceStatus.RUNNING: "Running",
            ServiceStatus.STOPPED: "Stopped",
            ServiceStatus.ERROR: "Error",
        }.get(status, "Unknown")

    def _get_status_class(self, status: ServiceStatus) -> str:
        return {
            ServiceStatus.RUNNING: "running",
            ServiceStatus.STOPPED: "stopped",
            ServiceStatus.ERROR: "error",
        }.get(status, "")

    def refresh_status(self):
        """Update the service status display."""
        service = self.manager.get_service(self.service_name)
        info = service.info()

        status_text = self._get_status_text(info.status)
        status_class = self._get_status_class(info.status)

        # Update status widget
        status_widget = self.query_one(f"#status-{self.service_name}", Static)
        icon = "●" if info.status == ServiceStatus.RUNNING else "○"
        status_widget.update(f"{icon} {status_text}")
        status_widget.set_classes(f"service-status {status_class}")

        # Update PID widget
        pid_widget = self.query_one(f"#pid-{self.service_name}", Static)
        pid_widget.update(f"PID: {info.pid}" if info.pid else "")

        # Update card border
        self.set_classes(status_class)

    def action_toggle(self):
        """Toggle service start/stop."""
        service = self.manager.get_service(self.service_name)
        info = service.info()

        if info.status == ServiceStatus.RUNNING:
            success, msg = service.stop()
        else:
            success, msg = service.start()

        self.app.notify(msg, severity="information" if success else "error")
        self.refresh_status()

    def action_restart(self):
        """Restart the service."""
        service = self.manager.get_service(self.service_name)
        success, msg = service.restart()
        self.app.notify(msg, severity="information" if success else "error")
        self.refresh_status()

    def action_logs(self):
        """Show logs for this service."""
        from syft_bg.tui.screens.log_screen import LogScreen

        self.app.push_screen(LogScreen(self.service_name))
