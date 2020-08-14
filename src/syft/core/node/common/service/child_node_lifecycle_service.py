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
from ...abstract.node import AbstractNode
from ...common.service.node_service import ImmediateNodeServiceWithoutReply
from ....common.serde.deserialize import _deserialize


class RegisterChildNodeMessage(ImmediateSyftMessageWithoutReply):
    def __init__(
        self, child_node_client_address: Address, address: Address, msg_id: UID = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.child_node_client_address = child_node_client_address

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> RegisterChildNodeMessage_PB:
        return RegisterChildNodeMessage_PB(
            child_node_client_address=self.child_node_client_address.serialize(),
            address=self.address.serialize(),
            msg_id=self.id.serialize(),
        )

    @staticmethod
    def _proto2object(proto: RegisterChildNodeMessage_PB) -> "RegisterChildNodeMessage":
        return RegisterChildNodeMessage(
            child_node_client_address=_deserialize(
                blob=proto.child_node_client_address
            ),
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
        # QUESTION: Now that these are serialized Address not Full Client
        # What do we want to store and which id do we want?
        # Is it target_id? Or one of the 4 locations?
        # It seems like the key is the "address" from a message so this needs to be
        # the intended future address, but which one?
        # Currently this is working:
        # msg.child_node_client_address.target_id.id
        """
        # old code
        # id=msg.child_node_client_address.id, data=msg.child_node_client_address,
        """
        obj_id = msg.child_node_client_address.target_id.id  # TODO: Fix, see above
        node.store.store(
            obj=StorableObject(id=obj_id, data=msg.child_node_client_address,)
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

        heritage_msg = HeritageUpdateMessage(
            new_ancestry_address=node.address, address=msg.child_node_client_address
        )

        location = msg.child_node_client_address.target_id.id

        try:
            in_memory_client = node.in_memory_client_registry[location]
            # we need to sign here with the current node not the destination side
            return in_memory_client.send_immediate_msg_without_reply(msg=heritage_msg)
        except Exception as e:
            print(f"{location} not on nodes in_memory_client. {e}")
            pass

        """"
        # old code had child_node_client which was a real object not a serialized address
        heritage_msg = HeritageUpdateMessage(
            new_ancestry_address=node, address=msg.child_node_client
        )

        msg.child_node_client.send_immediate_msg_without_reply(msg=heritage_msg)
        """

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[Type[RegisterChildNodeMessage]]:
        return [RegisterChildNodeMessage]
