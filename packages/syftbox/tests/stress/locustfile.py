import uuid
from pathlib import Path

from locust import FastHttpUser, between, task

import syftbox.client.exceptions
from syftbox.client.core import SyftBoxContext
from syftbox.client.plugins.sync.sync_action import ModifyRemoteAction
from syftbox.client.server_client import SyftBoxClient
from syftbox.lib.workspace import SyftWorkspace
from syftbox.server.sync.hash import hash_file
from syftbox.server.sync.models import FileMetadata

file_name = Path("loadtest.txt")


class SyftBoxUser(FastHttpUser):
    network_timeout = 5.0
    connection_timeout = 5.0
    wait_time = between(0.5, 1.5)

    def on_start(self):
        self.datasites = []
        self.email = "aziz@openmined.org"
        self.remote_state: dict[str, list[FileMetadata]] = {}

        self.syft_context = SyftBoxContext(
            email=self.email,
            client=SyftBoxClient(conn=self.client),
            workspace=SyftWorkspace(data_dir=Path(".")),
        )

        self.filepath = self.init_file()

    def init_file(self) -> Path:
        # create a file on local and send to server
        filepath = self.syft_context.datasite / file_name
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.touch()
        filepath.write_text(uuid.uuid4().hex)
        local_syncstate = hash_file(filepath.absolute(), root_dir=filepath.parent.absolute())
        try:
            self.syft_context.client.sync.create(local_syncstate.path, filepath.read_bytes())
        except syftbox.client.exceptions.SyftServerError:
            pass
        return filepath

    @task
    def sync_datasites(self):
        remote_datasite_states = self.sync_client.get_datasite_states()
        # logger.info(f"Syncing {len(remote_datasite_states)} datasites")
        all_files: list[FileMetadata] = []
        for remote_state in remote_datasite_states.values():
            all_files.extend(remote_state)

        all_paths = [f.path for f in all_files][:10]
        self.syft_context.client.sync.download_files_streaming(all_paths, self.syft_context.workspace.datasites)

    @task
    def apply_diff(self):
        self.filepath.write_text(uuid.uuid4().hex)
        local_syncstate = hash_file(self.filepath, root_dir=self.syft_context.datasite)
        remote_syncstate = self.syft_context.client.sync.get_metadata(self.filepath)

        action = ModifyRemoteAction(
            local_metadata=local_syncstate,
            remote_metadata=remote_syncstate,
        )
        action.execute(self.syft_context)

    @task
    def download(self):
        self.sync_client.download(self.filepath)
