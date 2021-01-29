# stdlib
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey

# syft relative
from .....decorators.syft_decorator_impl import syft_decorator
from .....logger import traceback_and_raise
from .....proto.core.node.common.action.save_object_pb2 import (
    SaveObjectAction as SaveObjectAction_PB,
)
from ....common.group import VerifyAll
from ....common.serde.deserialize import _deserialize
from ....common.serde.serializable import Serializable
from ....common.uid import UID
from ....io.address import Address
from ....store.storeable_object import StorableObject
from ...abstract.node import AbstractNode
from .common import ImmediateActionWithoutReply


class SaveObjectAction(ImmediateActionWithoutReply, Serializable):
    @syft_decorator(typechecking=True)
    def __init__(
        self,
        obj: StorableObject,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.obj = obj

    def execute_action(self, node: AbstractNode, verify_key: VerifyKey) -> None:
        self.obj.read_permissions = {
            node.verify_key: node.id,
            verify_key: None,  # we dont have the passed in sender's UID
        }
        node.store[self.obj.id] = self.obj

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> SaveObjectAction_PB:
        obj = self.obj._object2proto()
        addr = self.address.serialize()
        return SaveObjectAction_PB(
            obj=obj,
            address=addr
        )

    @staticmethod
    @syft_decorator(typechecking=True)
    def _proto2object(proto: SaveObjectAction_PB) -> "SaveObjectAction":
        obj = _deserialize(blob=proto.obj)
        addr = _deserialize(blob=proto.address)
        return SaveObjectAction(
            obj=obj,
            address=addr
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return SaveObjectAction_PB