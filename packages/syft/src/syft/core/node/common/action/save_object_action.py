# stdlib
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey

# relative
from .....proto.core.node.common.action.save_object_pb2 import (
    SaveObjectAction as SaveObjectAction_PB,
)
from ....common.serde.deserialize import _deserialize as deserialize
from ....common.serde.serializable import serializable
from ....common.serde.serialize import _serialize as serialize
from ....common.uid import UID
from ....io.address import Address
from ....store.storeable_object import StorableObject
from ...abstract.node import AbstractNode
from .common import ImmediateActionWithoutReply


@serializable()
class SaveObjectAction(ImmediateActionWithoutReply):
    def __init__(
        self,
        obj: StorableObject,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.obj = obj

    def __repr__(self) -> str:
        obj_str = str(self.obj)
        # make obj_str of reasonable length, if too long: cut into begin and end
        neg_index = max(-50, -len(obj_str) + 50)
        obj_str = obj_str = (
            obj_str[:50]
            if len(obj_str) < 50
            else obj_str[:50] + " ... " + obj_str[neg_index:]
        )
        return f"SaveObjectAction {obj_str}"

    def execute_action(self, node: AbstractNode, verify_key: VerifyKey) -> None:

        # If if there's another object with the same ID.
        node.store.check_collision(self.obj.id)

        self.obj.read_permissions = {
            node.verify_key: node.id,
            verify_key: None,  # we dont have the passed in sender's UID
        }
        self.obj.write_permissions = {
            node.verify_key: node.id,
            verify_key: None,  # we dont have the passed in sender's UID
        }
        node.store[self.obj.id] = self.obj

    def _object2proto(self) -> SaveObjectAction_PB:
        obj = self.obj._object2proto()
        addr = serialize(self.address)
        return SaveObjectAction_PB(obj=obj, address=addr)

    @staticmethod
    def _proto2object(proto: SaveObjectAction_PB) -> "SaveObjectAction":
        obj = deserialize(blob=proto.obj)
        addr = deserialize(blob=proto.address)
        return SaveObjectAction(obj=obj, address=addr)

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return SaveObjectAction_PB
