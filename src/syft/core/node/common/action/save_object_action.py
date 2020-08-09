from typing import Optional
from ...abstract.node import AbstractNode
from .common import ImmediateActionWithoutReply

from ....common.uid import UID
from ....io.address import Address
from ....common.serde.serializable import Serializable
from .....decorators.syft_decorator_impl import syft_decorator
from .....proto.core.node.common.action.save_object_pb2 import (
    SaveObjectAction as SaveObjectAction_PB,
)
from ....common.serde.deserialize import _deserialize
from ....store.storeable_object import StorableObject


class SaveObjectAction(ImmediateActionWithoutReply, Serializable):
    @syft_decorator(typechecking=True)
    def __init__(
        self, obj_id: UID, obj: object, address: Address, msg_id: Optional[UID] = None
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.obj_id = obj_id
        self.obj = obj

    def execute_action(self, node: AbstractNode) -> None:
        # save the object to the store
        node.store.store(obj=StorableObject(id=self.obj.id, data=self.obj)) # type: ignore

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> SaveObjectAction_PB:

        id_pb = self.obj_id.serialize()
        obj_ob = self.obj.serialize()  # type: ignore

        return SaveObjectAction(obj_id=id_pb, obj=obj_ob)

    @staticmethod
    @syft_decorator(typechecking=True)
    def _proto2object(proto: SaveObjectAction_PB) -> "SaveObjectAction":

        id = _deserialize(blob=proto.obj_id)
        obj = _deserialize(blob=proto.obj)

        return SaveObjectAction(obj_id=id, obj=obj)

    @staticmethod
    def get_protobuf_schema():
        return SaveObjectAction_PB
