from syft.core.common.message import ImmediateSyftMessageWithoutReply

from ...abstract.node import AbstractNode
from .common import ImmediateActionWithReply


class GetObjectResponseMessage(ImmediateSyftMessageWithoutReply):
    def __init__(self, obj, address, msg_id=None):
        super().__init__(address=address, msg_id=msg_id)
        self.obj = obj


class GetObjectAction(ImmediateActionWithReply):
    def __init__(self, obj_id, address, reply_to, msg_id=None):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.obj_id = obj_id

    def execute_action(self, node: AbstractNode) -> ImmediateSyftMessageWithoutReply:
        obj = node.store[self.obj_id]
        msg = GetObjectResponseMessage(obj=obj, address=self.reply_to, msg_id=None)

        # TODO: send EventualActionWithoutReply to delete the object at the node's
        # convenience instead of definitely having to delete it now
        del node.store[self.obj_id]
        return msg
