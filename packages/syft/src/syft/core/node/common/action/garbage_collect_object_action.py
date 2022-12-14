# # stdlib
# from typing import Optional

# # third party
# from google.protobuf.reflection import GeneratedProtocolMessageType
# from nacl.signing import VerifyKey

# # relative
# from .....logger import critical
# from .....proto.core.node.common.action.garbage_collect_object_pb2 import (
#     GarbageCollectObjectAction as GarbageCollectObjectAction_PB,
# )
# from ....common.serde.deserialize import _deserialize as deserialize
# from ....common.serde.serializable import serializable
# from ....common.serde.serialize import _serialize as serialize
# from ....common.uid import UID
# from ....io.address import Address
# from ...abstract.node import AbstractNode
# from .common import EventualActionWithoutReply


# @serializable()
# class GarbageCollectObjectAction(EventualActionWithoutReply):
#     def __init__(
#         self, id_at_location: UID, address: Address, msg_id: Optional[UID] = None
#     ):
#         super().__init__(address=address, msg_id=msg_id)
#         self.id_at_location = id_at_location

#     def execute_action(self, node: AbstractNode, verify_key: VerifyKey) -> None:
#         try:
#             node.store.delete(key=self.id_at_location)
#         except Exception as e:
#             critical(
#                 "> GarbageCollectObjectAction deletion exception "
#                 + f"{self.id_at_location} {e}"
#             )

#     def _object2proto(self) -> GarbageCollectObjectAction_PB:

#         id_pb = serialize(self.id_at_location)
#         addr = serialize(self.address)

#         return GarbageCollectObjectAction_PB(
#             id_at_location=id_pb,
#             address=addr,
#         )

#     @staticmethod
#     def _proto2object(
#         proto: GarbageCollectObjectAction_PB,
#     ) -> "GarbageCollectObjectAction":

#         id_at_location = deserialize(blob=proto.id_at_location)
#         addr = deserialize(blob=proto.address)

#         return GarbageCollectObjectAction(
#             id_at_location=id_at_location,
#             address=addr,
#         )

#     @staticmethod
#     def get_protobuf_schema() -> GeneratedProtocolMessageType:
#         return GarbageCollectObjectAction_PB
