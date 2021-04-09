"""A simple garbage collection heuritics."""
# third party
from typing_extensions import final

# syft relative
from ...node.common.action.garbage_collect_object_action import (
    GarbageCollectObjectAction,
)
from ..pointer import Pointer
from .gc_strategy import GCStrategy


@final
class GCSimple(GCStrategy):
    """The GCSimple Strategy."""

    def reap(self, pointer: Pointer) -> None:
        """
        Send a message to delete the remote object.

        Args:
            pointer (Pointer): Pointer that indicates to an object
                        that should get deleted

        Return:
            None
        """

        # Create the delete message
        msg = GarbageCollectObjectAction(
            id_at_location=pointer.id_at_location, address=pointer.client.address
        )

        # Send the message
        pointer.client.send_eventual_msg_without_reply(msg=msg)
