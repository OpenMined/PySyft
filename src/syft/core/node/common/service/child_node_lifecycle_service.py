# stdlib
from typing import List
from typing import Optional
from typing import Type

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from loguru import logger
from nacl.signing import VerifyKey

# syft relative
from .....decorators import syft_decorator
from .....proto.core.node.common.service.child_node_lifecycle_service_pb2 import (
    RegisterChildNodeMessage as RegisterChildNodeMessage_PB,
)
from ....common.message import ImmediateSyftMessageWithoutReply
from ....common.serde.deserialize import _deserialize
from ....common.uid import UID
from ....io.address import Address
from ....store.storeable_object import StorableObject
from ...abstract.node import AbstractNode
from ...common.service.node_service import ImmediateNodeServiceWithoutReply
from .auth import service_auth
from .heritage_update_service import HeritageUpdateMessage


class RegisterChildNodeMessage(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        lookup_id: UID,
        child_node_client_address: Address,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.lookup_id = lookup_id
        self.child_node_client_address = child_node_client_address

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> RegisterChildNodeMessage_PB:
        logger.debug(f"> {self.icon} -> Proto 🔢")
        return RegisterChildNodeMessage_PB(
            lookup_id=self.lookup_id.serialize(),  # TODO: not sure if this is needed anymore
            child_node_client_address=self.child_node_client_address.serialize(),
            address=self.address.serialize(),
            msg_id=self.id.serialize(),
        )

    @staticmethod
    def _proto2object(proto: RegisterChildNodeMessage_PB) -> "RegisterChildNodeMessage":
        msg = RegisterChildNodeMessage(
            lookup_id=_deserialize(blob=proto.lookup_id),
            child_node_client_address=_deserialize(
                blob=proto.child_node_client_address
            ),
            address=_deserialize(blob=proto.address),
            msg_id=_deserialize(blob=proto.msg_id),
        )
        logger.debug(f"> {msg.icon} <- 🔢 Proto")
        return msg

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return RegisterChildNodeMessage_PB


class ChildNodeLifecycleService(ImmediateNodeServiceWithoutReply):
    @classmethod
    def icon(cls) -> str:
        return "💾"

    @staticmethod
    @service_auth(root_only=True)
    @syft_decorator(typechecking=True)
    def process(
        node: AbstractNode, msg: RegisterChildNodeMessage, verify_key: VerifyKey
    ) -> None:
        logger.debug(
            f"> Executing {ChildNodeLifecycleService.pprint()} {msg.pprint} on {node.pprint}"
        )
        addr = msg.child_node_client_address
        lookup_id = msg.lookup_id  # TODO: Fix, see above

        node.store[lookup_id] = StorableObject(id=lookup_id, data=addr)

        logger.debug(
            (
                f"> Saving 💾 {addr.pprint} {addr.target_emoji()} with "
                + f"Key: {lookup_id} ➡️ {type(node.store)}"
            )
        )

        # Step 2: update the child node and its descendants with our node.id in their
        # .address objects

        # WARNING: This has no where to go
        # ---------------------------------
        # the goal of this code is to tell the child_node_client, hey you need to update
        # yourself with this new Address (which is up to 4 locations)
        # the old code didn't serialize the incoming object which was actually a real
        # python pointer to the original client
        # now that its a serialized address there are no pointers in memory to the
        # original child clients send_immediate_msg_without_reply function so
        # there is no way to invoke it
        logger.debug(
            f"> Sending 👪 Update from {node.pprint} back to {addr.target_emoji()}"
        )
        logger.debug("> Update Contains", type(node.address), node.address)
        heritage_msg = HeritageUpdateMessage(
            new_ancestry_address=node.address, address=msg.child_node_client_address
        )

        location = msg.child_node_client_address.target_id.id
        try:
            in_memory_client = node.in_memory_client_registry[location]
            # we need to sign here with the current node not the destination side
            in_memory_client.send_immediate_msg_without_reply(msg=heritage_msg)
            logger.debug(f"> Forwarding {msg.pprint} to {addr.target_emoji()}")
            return None
        except Exception as e:
            logger.error(f"{location} not on nodes in_memory_client. {e}")
            pass
        return None

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[Type[RegisterChildNodeMessage]]:
        return [RegisterChildNodeMessage]
