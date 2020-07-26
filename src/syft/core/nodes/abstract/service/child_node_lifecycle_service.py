from __future__ import annotations

from .....decorators import syft_decorator
from ...abstract.service.node_service import NodeService
from ...common.node import AbstractNode
from typing import List

from ....io.address import Address
from .....common.id import UID
from ....message.syft_message import SyftMessageWithoutReply
from ...common.node import AbstractNodeClient


class RegisterChildNodeMessage(SyftMessageWithoutReply):
    def __init__(
        self,
        child_node_client: AbstractNodeClient,
        address: Address,
        msg_id: UID = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.child_node_client = child_node_client


class ChildNodeLifecycleService(NodeService):
    @syft_decorator(typechecking=True)
    def process(self, node: AbstractNode, msg: RegisterChildNodeMessage) -> None:

        node.store.store_object(id=msg.child_node_client.target_node_id, obj=msg.child_node_client)

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        return [RegisterChildNodeMessage]
