"""A simple garbage collection heuritics."""
# future
from __future__ import annotations

# stdlib
from collections import defaultdict
from typing import Dict
from typing import List
from typing import TYPE_CHECKING

# syft relative
from ..common.uid import UID
from ..node.common.action.garbage_collect_batched_action import (
    GarbageCollectBatchedAction,
)
from ..pointer.pointer import Pointer
from .gc_strategy import GCStrategy

if TYPE_CHECKING:
    # syft relative
    from ..node.common.client import Client


class GCBatched(GCStrategy):
    """The GCBatched Strategy."""

    __slots__ = [
        "client_to_obj_ids",
        "threshold",
    ]

    client_to_obj_ids: Dict[Client, List[UID]]
    threshold: int

    def __init__(self, threshold: int = 10) -> None:
        """Construct the GCBatched Strategy.

        Args:
            threshold_client (int): the threshold after which a message
                would be sent to a client to delete all the objects
                that were recorded for them
        Return:
            None
        """
        self.client_to_obj_ids = defaultdict(list)
        self.threshold = threshold

    def reap(self, pointer: Pointer) -> None:
        """
        Send a simple message to delete the remote object.

        Args:
            pointer (Pointer): Pointe that should get deleted

        Return:
            None
        """

        client_objs = self.client_to_obj_ids[pointer.client]
        client_objs.append(pointer.id_at_location)

        nr_objs_client = len(client_objs)

        if nr_objs_client >= self.threshold:
            # Check the local threshold for the items that are kept for a client
            msg = GarbageCollectBatchedAction(
                ids_at_location=client_objs, address=pointer.client.address
            )

            pointer.client.send_eventual_msg_without_reply(msg)
            self.client_to_obj_ids.pop(pointer.client)

    def __del__(self) -> None:
        """Send a GarbageCollectBatchedAction to all the clients that are cached such
        that they delete all the items that should be deleted.
        """
        for client, ids_at_location in self.client_to_obj_ids.items():
            msg = GarbageCollectBatchedAction(
                ids_at_location=ids_at_location, address=client.address
            )

            client.send_eventual_msg_without_reply(msg)
