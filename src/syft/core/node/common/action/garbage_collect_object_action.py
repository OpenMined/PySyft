# stdlib
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from loguru import logger
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
    def __init__(
        self, id_at_location: UID, address: Address, msg_id: Optional[UID] = None
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.id_at_location = id_at_location

    def execute_action(self, node: AbstractNode, verify_key: VerifyKey) -> None:
        try:
            node.store.delete(key=self.id_at_location)
        except Exception as e:
            logger.critical(
                "> GarbageCollectObjectAction deletion exception "
                + f"{self.id_at_location} {e}"
            )

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> GarbageCollectObjectAction_PB:

        id_pb = self.id_at_location.serialize()
        addr = self.address.serialize()

        return GarbageCollectObjectAction_PB(
            id_at_location=id_pb,
            address=addr,
        )

    @staticmethod
    @syft_decorator(typechecking=True)
    def _proto2object(
        proto: GarbageCollectObjectAction_PB,
    ) -> "GarbageCollectObjectAction":

        id_at_location = _deserialize(blob=proto.id_at_location)
        addr = _deserialize(blob=proto.address)

        return GarbageCollectObjectAction(
            id_at_location=id_at_location,
            address=addr,
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return GarbageCollectObjectAction_PB
