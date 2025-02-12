from loguru import logger

from syftbox.client.base import SyftBoxContextInterface
from syftbox.client.plugins.sync.datasite_state import DatasiteState
from syftbox.client.plugins.sync.local_state import LocalState
from syftbox.client.plugins.sync.queue import SyncQueue, SyncQueueItem
from syftbox.client.plugins.sync.types import FileChangeInfo, SyncStatus


class SyncProducer:
    def __init__(self, context: SyftBoxContextInterface, queue: SyncQueue, local_state: LocalState):
        self.context = context
        self.queue = queue
        self.local_state = local_state

    def get_datasite_states(self) -> list[DatasiteState]:
        try:
            remote_datasite_states = self.context.client.sync.get_datasite_states()
        except Exception as e:
            logger.error(f"Failed to retrieve datasites from server, only syncing own datasite. Reason: {e}")
            remote_datasite_states = {}

        # Ensure we are always syncing own datasite
        if self.context.email not in remote_datasite_states:
            remote_datasite_states[self.context.email] = []

        datasite_states = [
            DatasiteState(self.context, email, remote_state=remote_state)
            for email, remote_state in remote_datasite_states.items()
        ]
        return datasite_states

    def add_ignored_to_local_state(self, datasite: DatasiteState) -> None:
        """
        NOTE: to keep logic simple, we do not remove ignored files from the local state here.
        Instead, they will be overwritten once the consumer processes the file.

        NOTE: To avoid spammy behaviour symlinks and hidden files are not included in the local state ignore list.
        Example: the symlinked apps .venv folders can contain 10k+ files
        """
        for path in datasite.get_syftignore_matches():
            prev_status_info = self.local_state.status_info.get(path, None)
            # Only add to local state if it's not already ignored previously
            is_ignored_previously = prev_status_info is not None and prev_status_info.status == SyncStatus.IGNORED
            if not is_ignored_previously:
                self.local_state.insert_status_info(path, SyncStatus.IGNORED)

    def enqueue_datasite_changes(self, datasite: DatasiteState) -> None:
        """
        Enqueue all out of sync files for the datasite,
        and track the ignored files in the local state.
        """
        try:
            datasite_changes = datasite.get_datasite_changes()

            if len(datasite_changes.permissions) or len(datasite_changes.files):
                logger.debug(
                    f"Enqueuing {len(datasite_changes.permissions)} permissions and {len(datasite_changes.files)} files for {datasite.email}"
                )
        except Exception as e:
            logger.error(f"Failed to get out of sync files for {datasite.email}. Reason: {e}")
            return
        for change in datasite_changes.permissions + datasite_changes.files:
            self.enqueue(change)

        self.add_ignored_to_local_state(datasite)

    def enqueue(self, change: FileChangeInfo) -> None:
        self.queue.put(SyncQueueItem(priority=change.get_priority(), data=change))
