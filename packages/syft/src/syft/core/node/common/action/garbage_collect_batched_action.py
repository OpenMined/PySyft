# stdlib
from typing import List
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey

# syft absolute
import syft as sy

# relative
from .....logger import critical
from .....proto.core.node.common.action.garbage_collect_batched_pb2 import (
    GarbageCollectBatchedAction as GarbageCollectBatchedAction_PB,
)
from ....common.serde.serializable import serializable
from ....common.uid import UID
from ....io.address import Address
from ...abstract.node import AbstractNode
from .common import EventualActionWithoutReply


@serializable()
class GarbageCollectBatchedAction(EventualActionWithoutReply):
    def __init__(
        self, ids_at_location: List[UID], address: Address, msg_id: Optional[UID] = None
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.ids_at_location = ids_at_location

    def execute_action(self, node: AbstractNode, verify_key: VerifyKey) -> None:
        try:
            for id_at_location in self.ids_at_location:
                node.store.delete(key=id_at_location)
        except Exception as e:
            critical(
                "> GarbageCollectBatchedAction deletion exception "
                + f"{id_at_location} {e}"
            )

    def _object2proto(self) -> GarbageCollectBatchedAction_PB:
        address = sy.serialize(self.address)
        res = GarbageCollectBatchedAction_PB(address=address)
        for id_obj in self.ids_at_location:
            res.ids_at_location.append(sy.serialize(id_obj))

        return res

    @staticmethod
    def _proto2object(
        proto: GarbageCollectBatchedAction_PB,
    ) -> "GarbageCollectBatchedAction":

        ids_at_location = []
        for id_at_location in proto.ids_at_location:
            ids_at_location.append(sy.deserialize(blob=id_at_location))
        addr = sy.deserialize(blob=proto.address)

        return GarbageCollectBatchedAction(
            ids_at_location=ids_at_location,
            address=addr,
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return GarbageCollectBatchedAction_PB
