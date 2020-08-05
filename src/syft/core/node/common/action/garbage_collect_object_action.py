from ...abstract.node import AbstractNode
from .common import EventualActionWithoutReply


class GarbageCollectObjectAction(EventualActionWithoutReply):
    def __init__(self, obj_id, address, msg_id=None):
        super().__init__(address=address, msg_id=msg_id)
        self.obj_id = obj_id

    def execute_action(self, node: AbstractNode) -> None:
        # TODO: make lazy
        # QUESTION: Where is delete_object defined
        node.store.delete_object(id=self.obj_id)
