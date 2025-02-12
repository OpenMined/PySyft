from typing import Any, Optional

import requests
from textual.containers import Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import Label, Markdown, Static

from syftbox import __version__
from syftbox.lib import Client
from syftbox.tui.widgets.logs_widget import SyftLogsWidget

INTRO_MD = """
### Welcome to SyftBox!

SyftBox is an innovative project by [OpenMined](https://openmined.org) that aims to make privacy-enhancing technologies (PETs) more accessible and user-friendly for developers. It provides a modular and intuitive framework for building PETs applications with minimal barriers, regardless of the programming language or environment.

### Important Resources
- 📚 Check the docs at https://syftbox-documentation.openmined.org/
- 📊 View the [Stats Dashboard](https://syftbox.openmined.org/datasites/andrew@openmined.org/stats.html)
- 🔧 View our [GitHub Repository](https://github.com/OpenMined/syft)
- 🔍 Browse [Available Datasets](https://syftbox.openmined.org/datasites/aggregator@openmined.org/data_search/)

Need help? Join us on [Slack](https://slack.openmined.org/) 💬
"""


class StatusDashboard(Widget):
    DEFAULT_CSS = """
    StatusDashboard {
        height: auto;
    }
    """

    def __init__(
        self,
        syftbox_context: Client,
        *,
        classes: Optional[str] = None,
    ):
        self.syftbox_context = syftbox_context
        super().__init__(
            classes=classes,
        )

    def compose(self) -> Any:
        yield Static("[blue]Status[/blue]\n")
        server_url = f"[link={self.syftbox_context.config.server_url}]{self.syftbox_context.config.server_url}[/link]"
        client_url = f"[link={self.syftbox_context.config.client_url}]{self.syftbox_context.config.client_url}[/link]"
        data_dir = (
            f"[link=file://{self.syftbox_context.workspace.data_dir}]{self.syftbox_context.workspace.data_dir}[/link]"
        )
        yield Static(f"Syftbox version: [green]{__version__}[/green]")
        yield Static(f"User: [green]{self.syftbox_context.email}[/green]")
        yield Static(f"Syftbox folder: [green]{data_dir}[/green]")
        yield Static(f"Server URL: [green]{server_url}[/green]")
        yield Static(f"Local URL: [green]{client_url}[/green]")

        sync_status = "🟢 [green]Active[/green]" if self._sync_is_alive() else "🔴 [red]Inactive[/red]"
        yield Label(f"Sync: {sync_status}", id="sync_status")

        apps_count = self.count_apps()
        apps_color = "green" if apps_count > 0 else "red"
        yield Label(f"Installed APIs: [{apps_color}]{apps_count}[/{apps_color}]", id="api_count")

        self.set_interval(1, self.update_values)

    def update_values(self) -> None:
        sync_status_widget = self.query_exactly_one("#sync_status", expect_type=Label)
        api_count_widget = self.query_exactly_one("#api_count", expect_type=Label)

        sync_status = "🟢 [green]Active[/green]" if self._sync_is_alive() else "🔴 [red]Inactive[/red]"
        sync_status_widget.content = f"Sync: {sync_status}"

        apps_count = self.count_apps()
        apps_color = "green" if apps_count > 0 else "red"
        api_count_widget.content = f"Installed APIs: [{apps_color}]{apps_count}[/{apps_color}]"

        sync_status_widget.refresh()
        api_count_widget.refresh()

    def _sync_is_alive(self) -> bool:
        try:
            response = requests.get(f"{self.syftbox_context.config.client_url}/sync/health")
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def count_apps(self) -> int:
        api_dir = self.syftbox_context.workspace.apps
        return len([d for d in api_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])


class HomeWidget(Widget):
    DEFAULT_CSS = """
    HomeWidget {
        height: auto;
    }
    """

    def __init__(self, syftbox_context: Client) -> None:
        super().__init__()
        self.syftbox_context = syftbox_context
        self.info_widget = Markdown(INTRO_MD, classes="info")
        self.logs_widget = SyftLogsWidget(
            syftbox_context=self.syftbox_context,
            endpoint="/logs",
            title="SyftBox Logs",
            refresh_every=2,
            classes="syftbox-logs",
        )

    def compose(self) -> Any:
        with Horizontal():
            yield StatusDashboard(self.syftbox_context, classes="status")
            yield Vertical(
                self.info_widget,
                self.logs_widget,
                classes="main",
            )
