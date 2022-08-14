# stdlib
from typing import Optional

# third party
from nacl.signing import VerifyKey

# relative
from .....logger import critical
from ....common.serde.serializable import serializable
from ....common.uid import UID
from ....io.address import Address
from ...abstract.node import AbstractNode
from .common import EventualActionWithoutReply


@serializable(recursive_serde=True)
class GarbageCollectObjectAction(EventualActionWithoutReply):
    __attr_allowlist__ = ["address", "id_at_location", "id"]

    def __init__(
        self, id_at_location: UID, address: Address, msg_id: Optional[UID] = None
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.id_at_location = id_at_location

    def execute_action(self, node: AbstractNode, verify_key: VerifyKey) -> None:
        try:
            node.store.delete(key=self.id_at_location)
        except Exception as e:
            critical(
                "> GarbageCollectObjectAction deletion exception "
                + f"{self.id_at_location} {e}"
            )
