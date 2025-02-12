import urllib
import urllib.parse
from typing import Any, List

from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Label, ListItem, ListView

from syftbox.lib import Client
from syftbox.tui.widgets.logs_widget import SyftLogsWidget


class APIWidget(Widget):
    DEFAULT_CSS = """
    APIWidget {
        height: auto;
    }
    """

    def __init__(
        self,
        syftbox_context: Client,
    ):
        super().__init__()
        self.syftbox_context = syftbox_context
        self.apps: List[str] = []

    def compose(self) -> Any:
        self.apps = self.get_installed_apps()

        with Horizontal():
            list_view = ListView(*[ListItem(Label(app), id=app) for app in self.apps], classes="sidebar")
            list_view.styles.width = "20%"
            yield list_view

            self.log_widget = SyftLogsWidget(
                self.syftbox_context, None, title="API Logs", refresh_every=2, classes="api-logs"
            )
            self.set_app_logs(self.apps[0])

            yield self.log_widget

    def set_app_logs(self, app_name: str) -> None:
        """Update the logs widget to show logs for the given app."""
        app_name = urllib.parse.quote(app_name)
        endpoint = f"/apps/logs/{app_name}"
        self.log_widget.endpoint = endpoint
        self.log_widget.refresh_logs()

    def get_installed_apps(self) -> list[str]:
        api_dir = self.syftbox_context.workspace.apps
        return [d.name for d in api_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        app_name = event.item.id
        self.set_app_logs(app_name)
