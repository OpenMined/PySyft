from pathlib import Path
from typing import Any

from textual.app import App
from textual.widgets import Footer, Header, TabbedContent, TabPane

from syftbox.lib import Client
from syftbox.tui.widgets.api_widget import APIWidget
from syftbox.tui.widgets.datasites_widget import DatasiteSelector
from syftbox.tui.widgets.home_widget import HomeWidget
from syftbox.tui.widgets.sync_widget import SyncWidget


class SyftBoxTUI(App):
    CSS_PATH = Path(__file__).parent.parent / "assets" / "tui.tcss"
    BINDINGS = [
        ("h", "switch_tab('Home')", "Home"),
        ("a", "switch_tab('APIs')", "APIs"),
        ("d", "switch_tab('Datasites')", "Datasites"),
        ("s", "switch_tab('Sync')", "Sync"),
        ("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        syftbox_context: Client,
    ):
        super().__init__()
        self.syftbox_context = syftbox_context

    def action_switch_tab(self, tab: str) -> None:
        self.query_one(TabbedContent).active = tab

    def on_mount(self) -> None:
        self.title = "SyftBox"

    def compose(self) -> Any:
        yield Header(name="SyftBox")
        with TabbedContent():
            with TabPane("Home", id="Home"):
                yield HomeWidget(self.syftbox_context)
            with TabPane("APIs", id="APIs"):
                yield APIWidget(self.syftbox_context)
            with TabPane("Datasites", id="Datasites"):
                yield DatasiteSelector(
                    base_path=self.syftbox_context.workspace.datasites,
                    default_datasite=self.syftbox_context.email,
                )
            with TabPane("Sync", id="Sync"):
                yield SyncWidget(self.syftbox_context)
        yield Footer()


# config = SyftClientConfig.load()
# syftbox_context = Client(config)
# app = SyftBoxTUI(syftbox_context)
