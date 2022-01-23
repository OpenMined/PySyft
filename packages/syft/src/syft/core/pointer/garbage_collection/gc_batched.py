"""A batched garbage collection heuristics.

Deleting a remote object using this logic might not send a message indicating
to the Data Owner to remove the object from the store.

Instead, it would collect the id of that object to be deleted and would only
send the "delete message" when a specific threshold limit is reached. At this
point the Data Owner will loop through all the ids from the received message and
will remove each object from the store.
"""
# stdlib
from typing import List
from typing import Optional
from typing import TYPE_CHECKING

# third party
from typing_extensions import final

# relative
from ...common.uid import UID
from ...node.common.action.garbage_collect_batched_action import (
    GarbageCollectBatchedAction,
)
from ..pointer import Pointer
from .gc_strategy import GCStrategy

if TYPE_CHECKING:

    # relative
    from ...node.common.client import Client


@final
class GCBatched(GCStrategy):
    """The GCBatched Strategy."""

    __slots__ = [
        "obj_ids",
        "threshold",
        "client",
    ]

    client: Optional["Client"]
    obj_ids: List[UID]
    threshold: int

    def __init__(self, threshold: int = 10) -> None:
        """Construct the GCBatched Strategy.

        Args:
            threshold (int): the threshold after which a message
                would be sent to delete all the objects that were cached
        Return:
            None
        """
        self.obj_ids = []
        self.threshold = threshold
        self.client = None

    def reap(self, pointer: Pointer) -> None:
        """
        Check if we passed the threshold of objects that we should cache.
        If yes, then send a message to delete the objects.
        If no, cache the object id to be deleted when there are more ids
        collected.

        Args:
            pointer (Pointer): Pointer to the object that should get deleted

        Return:
            None
        """

        self.obj_ids.append(pointer.id_at_location)

        nr_objs_client = len(self.obj_ids)

        if nr_objs_client >= self.threshold:
            # Check the local threshold for the items that are kept for a client
            msg = GarbageCollectBatchedAction(
                ids_at_location=self.obj_ids, address=pointer.client.address
            )

            pointer.client.send_eventual_msg_without_reply(msg)
            self.obj_ids = []

        self.client = pointer.client

    def __del__(self) -> None:
        """Send a GarbageCollectBatchedAction to the client such that all the
        objects that are cached to be deleted would be deleted.
        """
        if self.client is None:
            return

        msg = GarbageCollectBatchedAction(
            ids_at_location=self.obj_ids, address=self.client.address
        )

        self.client.send_eventual_msg_without_reply(msg)
