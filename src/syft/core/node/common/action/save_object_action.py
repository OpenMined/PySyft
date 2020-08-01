from ...abstract.node import AbstractNode
from .common import ImmediateActionWithoutReply


class SaveObjectAction(ImmediateActionWithoutReply):
    def __init__(self, obj_id, obj, address, msg_id=None):
        super().__init__(address=address, msg_id=msg_id)
        self.obj_id = obj_id
        self.obj = obj

    def execute_action(self, node: AbstractNode):
        # save the object to the store
        node.store.store_object(id=self.obj_id, obj=self.obj)
