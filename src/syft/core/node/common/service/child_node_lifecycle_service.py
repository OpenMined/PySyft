from typing import List, Type

from syft.core.common.message import ImmediateSyftMessageWithoutReply
from syft.core.common.uid import UID

from .....decorators import syft_decorator
from ....io.address import Address
from ...abstract.node import AbstractNode, AbstractNodeClient
from ...common.service.node_service import ImmediateNodeServiceWithoutReply
from .heritage_update_service import HeritageUpdateMessage
from ....store.storeable_object import StorableObject


class RegisterChildNodeMessage(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        child_node_client: AbstractNodeClient,
        address: Address,
        msg_id: UID = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.child_node_client = child_node_client


class ChildNodeLifecycleService(ImmediateNodeServiceWithoutReply):
    @syft_decorator(typechecking=True)
    def process(self, node: AbstractNode, msg: RegisterChildNodeMessage) -> None:

        # Step 1: Store the client to the child in our object store.
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
