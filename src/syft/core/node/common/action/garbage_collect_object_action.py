# stdlib
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey

# syft relative
from .....decorators.syft_decorator_impl import syft_decorator
from .....proto.core.node.common.action.garbage_collect_object_pb2 import (
    GarbageCollectObjectAction as GarbageCollectObjectAction_PB,
)
from ....common.serde.deserialize import _deserialize
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
        try:
            del node.store[self.obj_id]
        except KeyError:
            # This might happen when we finish running our code/notebook
            # The objects might have already been deleated
            pass

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> GarbageCollectObjectAction_PB:

        id_pb = self.obj_id.serialize()
        addr = self.address.serialize()

        return GarbageCollectObjectAction_PB(
            obj_id=id_pb,
            address=addr,
        )

    @staticmethod
    @syft_decorator(typechecking=True)
    def _proto2object(
        proto: GarbageCollectObjectAction_PB,
    ) -> "GarbageCollectObjectAction":

        id = _deserialize(blob=proto.obj_id)
        addr = _deserialize(blob=proto.address)

        return GarbageCollectObjectAction(
            obj_id=id,
            address=addr,
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return GarbageCollectObjectAction_PB
