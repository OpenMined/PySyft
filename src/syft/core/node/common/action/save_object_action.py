from typing import Optional
from ...abstract.node import AbstractNode
from .common import ImmediateActionWithoutReply

from syft.core.common.uid import UID
from syft.core.io.address import Address


class SaveObjectAction(ImmediateActionWithoutReply):
    def __init__(
        self, obj_id: int, obj: object, address: Address, msg_id: Optional[UID] = None
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.obj_id = obj_id
        self.obj = obj

    def execute_action(self, node: AbstractNode) -> None:
        # save the object to the store
        node.store.store_object(id=self.obj_id, obj=self.obj)
