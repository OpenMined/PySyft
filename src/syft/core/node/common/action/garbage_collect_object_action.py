# stdlib
from typing import Optional

# third party
from nacl.signing import VerifyKey

# syft relative
from ....common.uid import UID
from ....io.address import Address
from ...abstract.node import AbstractNode
from .common import EventualActionWithoutReply


class GarbageCollectObjectAction(EventualActionWithoutReply):
    def __init__(self, obj_id: UID, address: Address, msg_id: Optional[UID] = None):
        super().__init__(address=address, msg_id=msg_id)
        self.obj_id = obj_id

    def execute_action(self, node: AbstractNode, verify_key: VerifyKey) -> None:
        # TODO: make lazy
        # QUESTION: Where is delete_object defined
        del node.store[self.obj_id]
