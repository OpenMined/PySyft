"""TUI dashboard for SyftBox background services."""

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Footer, Header, Static

from syft_bg.services import ServiceManager
from syft_bg.tui.widgets import ActivityFeed, ServiceCard


class SyftBgApp(App):
    """TUI dashboard for SyftBox background services."""

    TITLE = "SyftBox Background Services"
    SUB_TITLE = "Press ? for help"

    CSS = """
    Screen {
        background: $background;
    }

    #main-container {
        width: 100%;
        height: 100%;
        padding: 1 2;
    }

    #top-row {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }

    #services-column {
        width: 1fr;
        height: auto;
    }

    .title {
        text-style: bold;
        margin-bottom: 1;
        color: $primary;
    }

    #activity-container {
        width: 100%;
        height: 1fr;
        min-height: 10;
    }

    #status-bar {
        dock: bottom;
        height: 1;
        background: $surface;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("question_mark", "help", "Help", key_display="?"),
        Binding("a", "start_all", "Start All"),
        Binding("x", "stop_all", "Stop All"),
        Binding("R", "refresh", "Refresh"),
        Binding("i", "init", "Init"),
    ]

    def __init__(self):
        super().__init__()
        self.manager = ServiceManager()

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="main-container"):
            with Horizontal(id="top-row"):
                with Vertical(id="services-column"):
                    yield Static("SERVICES", classes="title")
                    for service_name in self.manager.list_services():
                        yield ServiceCard(service_name, id=f"card-{service_name}")
            with Container(id="activity-container"):
                yield ActivityFeed(id="activity-feed")
        yield Footer()

    def action_start_all(self):
        """Start all services."""
        results = self.manager.start_all()
        for name, (success, msg) in results.items():
            self.notify(
                f"{name}: {msg}", severity="information" if success else "error"
            )
        self._refresh_all()

    def action_stop_all(self):
        """Stop all services."""
        results = self.manager.stop_all()
        for name, (success, msg) in results.items():
            self.notify(
                f"{name}: {msg}", severity="information" if success else "error"
            )
        self._refresh_all()

    def action_refresh(self):
        """Refresh all service statuses."""
        self._refresh_all()
        self.notify("Status refreshed")

    def action_init(self):
        """Run the init flow (exits TUI first)."""
        self.exit(return_code=2)  # Special code for init

    def action_help(self):
        """Show help."""
        help_text = """
Keyboard Shortcuts:
  Up/Down - Navigate services
  Enter   - Start/Stop selected service
  r       - Restart selected service
  l       - View service logs
  a       - Start all services
  x       - Stop all services
  R       - Refresh status
  i       - Run init setup
  q       - Quit
        """
        self.notify(help_text.strip(), timeout=10)

    def _refresh_all(self):
        """Refresh status of all service cards and activity feed."""
        for service_name in self.manager.list_services():
            card = self.query_one(f"#card-{service_name}", ServiceCard)
            card.refresh_status()

        try:
            feed = self.query_one("#activity-feed", ActivityFeed)
            feed.refresh_activity()
        except Exception:
            pass

    def on_mount(self):
        """Set up auto-refresh timer."""
        self.set_interval(5, self._refresh_all)


def run_tui():
    """Run the TUI dashboard."""
    app = SyftBgApp()
    result = app.run()

    # Handle special exit codes
    if result == 2:
        from syft_bg.cli.init import InitFlowError, run_init_flow

        try:
            run_init_flow()
        except InitFlowError as e:
            print(f"Error: {e}")
