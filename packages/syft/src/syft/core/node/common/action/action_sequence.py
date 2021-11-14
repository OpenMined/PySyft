# stdlib
from typing import List
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey

# syft absolute
import syft as sy

# relative
from .....proto.core.node.common.action.action_sequence_pb2 import (
    ActionSequence as ActionSequence_PB,
)
from ....common.serde.serializable import serializable
from ....common.uid import UID
from ....io.address import Address
from ...abstract.node import AbstractNode
from .common import ImmediateActionWithoutReply
from .save_object_action import SaveObjectAction


@serializable()
class ActionSequence(ImmediateActionWithoutReply):
    def __init__(
        self,
        obj_lst: List[SaveObjectAction],
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.obj_lst = obj_lst

    def __repr__(self) -> str:
        obj_str = str(self.obj_lst)
        # make obj_str of reasonable length, if too long: cut into begin and end
        neg_index = max(-50, -len(obj_str) + 50)
        obj_str = obj_str = (
            obj_str[:50]
            if len(obj_str) < 50
            else obj_str[:50] + " ... " + obj_str[neg_index:]
        )
        return f"ActionSequence {obj_str}"

    def execute_action(self, node: AbstractNode, verify_key: VerifyKey) -> None:
        for obj in self.obj_lst:
            obj.execute_action(node=node, verify_key=verify_key)

    def _object2proto(self) -> ActionSequence_PB:
        obj_lst = list(map(lambda x: sy.serialize(x), self.obj_lst))
        addr = sy.serialize(self.address)
        return ActionSequence_PB(obj=obj_lst, address=addr)

    @staticmethod
    def _proto2object(proto: ActionSequence_PB) -> "ActionSequence":
        obj_lst = list(map(lambda x: sy.deserialize(blob=x), proto.obj))
        addr = sy.deserialize(blob=proto.address)
        return ActionSequence(obj_lst=obj_lst, address=addr)

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return ActionSequence_PB
