# external class imports
from nacl.signing import VerifyKey
from typing import List
from typing import Type
from google.protobuf.reflection import GeneratedProtocolMessageType

from syft.core.common.message import ImmediateSyftMessageWithoutReply
from syft.core.common.uid import UID

from .....proto.core.node.common.service.child_node_lifecycle_service_pb2 import (
    RegisterChildNodeMessage as RegisterChildNodeMessage_PB,
)
from .auth import service_auth
from ....io.address import Address
from .....decorators import syft_decorator
from ....store.storeable_object import StorableObject
from .heritage_update_service import HeritageUpdateMessage
from ...abstract.node import AbstractNode, AbstractNodeClient
from ...common.service.node_service import ImmediateNodeServiceWithoutReply
from ....common.serde.deserialize import _deserialize


class RegisterChildNodeMessage(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        child_node_client: AbstractNodeClient,
        address: Address,
        msg_id: UID = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.child_node_client = child_node_client

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> RegisterChildNodeMessage_PB:
        return RegisterChildNodeMessage_PB(
            child_node_client=self.child_node_client.serialize(),
            address=self.address.serialize(),
            msg_id=self.id.serialize(),
        )

    @staticmethod
    def _proto2object(proto: RegisterChildNodeMessage_PB) -> "RegisterChildNodeMessage":
        return RegisterChildNodeMessage(
            child_node_client=_deserialize(blob=proto.child_node_client),
            address=_deserialize(blob=proto.address),
            msg_id=_deserialize(blob=proto.msg_id),
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return RegisterChildNodeMessage_PB


class ChildNodeLifecycleService(ImmediateNodeServiceWithoutReply):
    @staticmethod
    @service_auth(root_only=True)
    @syft_decorator(typechecking=True)
    def process(
        node: AbstractNode, msg: RegisterChildNodeMessage, verify_key: VerifyKey
    ) -> None:
        # Step 1: Store the client to the child in our object store.
        # QUESTION: This seems to be where the auth stuff unravels as the entire chain
        # of references gets serialized and cant be represented as a cyclic proto
        node.store.store(
            obj=StorableObject(id=msg.child_node_client.id, data=msg.child_node_client)
        )

        # Step 2: update the child node and its descendants with our node.id in their
        # .address objects
        heritage_msg = HeritageUpdateMessage(
            new_ancestry_address=node, address=msg.child_node_client
        )

        msg.child_node_client.send_immediate_msg_without_reply(msg=heritage_msg)

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[Type[RegisterChildNodeMessage]]:
        return [RegisterChildNodeMessage]
