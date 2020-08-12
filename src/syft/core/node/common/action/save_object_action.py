# external class imports
from typing import Optional
from nacl.signing import VerifyKey
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft imports
from ....common.uid import UID
from ....io.address import Address
from ...abstract.node import AbstractNode
from .common import ImmediateActionWithoutReply
from ....store.storeable_object import StorableObject
from ....common.serde.deserialize import _deserialize
from ....common.serde.serializable import Serializable
from .....decorators.syft_decorator_impl import syft_decorator
from .....proto.core.node.common.action.save_object_pb2 import (
    SaveObjectAction as SaveObjectAction_PB,
)


class SaveObjectAction(ImmediateActionWithoutReply, Serializable):
    @syft_decorator(typechecking=True)
    def __init__(
        self, obj_id: UID, obj: object, address: Address, msg_id: Optional[UID] = None
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.obj_id = obj_id
        self.obj = obj

    def execute_action(self, node: AbstractNode, verify_key: VerifyKey) -> None:
        # save the object to the store
        node.store.store(obj=StorableObject(id=self.obj.id,
                                            data=self.obj,
                                            read_permissions=set([verify_key, node.verify_key])))  # type: ignore

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> SaveObjectAction_PB:

        id_pb = self.obj_id.serialize()
        obj_ob = self.obj.serialize()  # type: ignore
        addr = self.address.serialize()

        return SaveObjectAction_PB(obj_id=id_pb, obj=obj_ob, address=addr)

    @staticmethod
    @syft_decorator(typechecking=True)
    def _proto2object(proto: SaveObjectAction_PB) -> "SaveObjectAction":

        id = _deserialize(blob=proto.obj_id)
        obj = _deserialize(blob=proto.obj)
        addr = _deserialize(blob=proto.address)

        return SaveObjectAction(obj_id=id, obj=obj, address=addr)

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return SaveObjectAction_PB
