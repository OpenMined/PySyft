"""Log screen for viewing service logs."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, ScrollableContainer
from textual.screen import ModalScreen
from textual.widgets import Static

from syft_bg.services import ServiceManager


class LogScreen(ModalScreen):
    """Screen showing logs for a service."""

    DEFAULT_CSS = """
    LogScreen {
        align: center middle;
    }

    #log-dialog {
        width: 90%;
        height: 90%;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    .log-header {
        text-style: bold;
        margin-bottom: 1;
        color: $primary;
    }

    .log-content {
        width: 100%;
        height: 100%;
        overflow-y: auto;
        background: $background;
        padding: 1;
    }

    .log-hint {
        margin-top: 1;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("escape", "pop_screen", "Back"),
        Binding("q", "pop_screen", "Back"),
    ]

    def __init__(self, service_name: str):
        super().__init__()
        self.service_name = service_name
        self.manager = ServiceManager()

    def compose(self) -> ComposeResult:
        logs = self.manager.get_logs(self.service_name, lines=100)
        log_text = "\n".join(logs) if logs else "No logs available"

        with Container(id="log-dialog"):
            yield Static(f"LOGS: {self.service_name.upper()}", classes="log-header")
            yield ScrollableContainer(
                Static(log_text, classes="log-text"),
                classes="log-content",
            )
            yield Static("Press ESC or q to close", classes="log-hint")

    def action_pop_screen(self):
        self.app.pop_screen()
