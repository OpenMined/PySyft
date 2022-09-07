# stdlib
from typing import List
from typing import Optional

# third party
from nacl.signing import VerifyKey

# relative
from .....logger import critical
from ....common.serde.serializable import serializable
from ....common.uid import UID
from ....io.address import Address
from ...abstract.node import AbstractNode
from .common import ImmediateActionWithoutReply


@serializable(recursive_serde=True)
class GarbageCollectBatchedAction(ImmediateActionWithoutReply):
    __attr_allowlist__ = ["ids_at_location", "address", "id"]

    def __init__(
        self, ids_at_location: List[UID], address: Address, msg_id: Optional[UID] = None
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.ids_at_location = ids_at_location

    def execute_action(self, node: AbstractNode, verify_key: VerifyKey) -> None:
        try:
            for id_at_location in self.ids_at_location:
                node.store.delete(key=id_at_location)
        except Exception as e:
            critical(
                "> GarbageCollectBatchedAction deletion exception "
                + f"{id_at_location} {e}"
            )
