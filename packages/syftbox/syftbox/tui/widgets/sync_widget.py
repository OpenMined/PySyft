from typing import Any, List, Optional

import requests
from textual.binding import Binding
from textual.containers import ScrollableContainer
from textual.widgets import DataTable, Input, Label, Static

from syftbox.client.plugins.sync.local_state import SyncStatusInfo
from syftbox.lib import Client


class SyncWidget(Static):
    LIMIT = 100
    BINDINGS = [
        Binding("f", "focus_search", "Search", show=True),
    ]

    def __init__(self, syftbox_context: Client) -> None:
        super().__init__()
        self.syftbox_context = syftbox_context

    def compose(self) -> Any:
        yield Label("Filter Files")
        yield Input(
            placeholder="Enter a glob pattern to filter files",
            id="path_filter",
        )

        self.table = DataTable()
        self.table.add_columns("Path", "Status", "Action", "Last Update", "Message")
        yield Label("Sync Events", classes="padding-top")
        yield ScrollableContainer(self.table)
        yield Static(f"Showing last {self.LIMIT} sync events", classes="dim")

    def on_mount(self) -> None:
        self._refresh_table()

    def _refresh_table(self, path_filter: Optional[str] = None) -> None:
        self.table.clear()

        status_info = self._get_sync_state(path_filter)
        for info in status_info:
            self.table.add_row(
                str(info.path),
                info.status.value,
                info.action.value if info.action else "",
                info.timestamp.astimezone().strftime("%Y-%m-%d %H:%M:%S"),
                info.message or "",
            )

    def _get_sync_state(self, path_glob: Optional[str] = None) -> List[SyncStatusInfo]:
        try:
            response = requests.get(
                f"{self.syftbox_context.config.client_url}/sync/state",
                params={"path_glob": path_glob, "limit": self.LIMIT},
            )
            response.raise_for_status()
            return [SyncStatusInfo(**item) for item in response.json()]
        except Exception:
            return []

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "path_filter":
            self._refresh_table(event.value)

    def action_focus_search(self) -> None:
        self.query_one(Input).focus()
