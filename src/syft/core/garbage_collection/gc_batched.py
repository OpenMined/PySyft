"""A simple garbage collection heuritics."""
# stdlib
from collections import defaultdict
from typing import Dict
from typing import List

# syft relative
from ..common.uid import UID
from ..node.abstract.node import AbstractNodeClient
from ..node.common.action.garbage_collect_batched_action import (
    GarbageCollectBatchedAction,
)
from ..pointer.pointer import Pointer
from .gc_strategy import GCStrategy


class GCBatched(GCStrategy):
    """The GCBatched Strategy."""

    __slots__ = [
        "client_to_obj_ids",
        "threshold_client",
        "threshold_total",
        "count_total",
    ]

    client_to_obj_ids: Dict[AbstractNodeClient, List[UID]]
    threshold_client: int
    threshold_total: int
    count_total: int

    def __init__(self, threshold_client: int = 10, threshold_total: int = 50) -> None:
        """Construct the GCBatched Strategy.

        Args:
            threshold_client (int): the threshold after which a message
                would be sent to a client to delete all the objects
                that were recorded for them
            threshold_total (int): the threshold after which a message
                would be sent to all clients to delete all the objects
                that were recorded

        Return:
            None
        """
        self.client_to_obj_ids = defaultdict(list)
        self.threshold_client = threshold_client
        self.threshold_total = threshold_total
        self.count_total = 0

    def __clear_all_objects(self) -> None:
        """Send a GarbageCollectBatchedAction to all the clients that are cached such
        that they delete all the items that should be deleted.
        """
        for client, ids_at_location in self.client_to_obj_ids.items():
            msg = GarbageCollectBatchedAction(
                ids_at_location=ids_at_location, address=client.address
            )

            client.send_eventual_msg_without_reply(msg)

        self.count_total = 0
        self.client_to_obj_ids.clear()

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
        self.count_total += 1

        if self.count_total >= self.threshold_total:
            # Check the global threshold for all items that are kept
            self.__clear_all_objects()
            return

        nr_objs_client = len(client_objs)
        if nr_objs_client >= self.threshold_client:
            # Check the local threshold for the items that are kept for a client
            msg = GarbageCollectBatchedAction(
                ids_at_location=client_objs, address=pointer.client.address
            )

            pointer.client.send_eventual_msg_without_reply(msg)

            self.count_total -= nr_objs_client
            self.client_to_obj_ids.pop(pointer.client)

    def force_gc(self) -> None:
        """Force the garbage collection of all items using this strategy."""
        self.__clear_all_objects()
