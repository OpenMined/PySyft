"""The purpose of this service is to inform lower level devices
of changes in the hierarchy above them. For example, if a Domain
registers within a new Network or if a Device registers within
a new Domain, all the other child node will need to know this
information to populate complete addresses into their clients."""

from typing import List

from syft.core.common.message import ImmediateSyftMessageWithoutReply
from syft.core.common.uid import UID

from .....decorators import syft_decorator
from ....io.address import Address
from ...abstract.node import AbstractNode
from .node_service import ImmediateNodeServiceWithoutReply

# TODO: change all old_message names in syft to have "WithReply" or "WithoutReply"
# at the end of the name


class HeritageUpdateMessage(ImmediateSyftMessageWithoutReply):
    def __init__(
        self, new_ancestry_address: Address, address: Address, msg_id: UID = None
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.new_ancestry_address = new_ancestry_address


class HeritageUpdateService(ImmediateNodeServiceWithoutReply):
    @syft_decorator(typechecking=True)
    def process(self, node: AbstractNode, msg: HeritageUpdateMessage) -> None:
        print(f"Updating to {msg.new_ancestry_address} on note {node}")
        addr = msg.new_ancestry_address
        if addr.network is not None:
            node.network = addr.network
        if addr.domain is not None:
            node.domain = addr.domain
        if addr.device is not None:
            node.device = addr.device

        # TODO: solve this with node group address?
        print("child nodes of:" + str(node.name))
        for node_client in node.known_child_nodes:
            print("\t" + str(node_client.data.name))
            # TODO: Client (and possibly Node) should subclass from StorableObject
            msg.address = node_client.data
            node_client.data.send_immediate_msg_without_reply(msg=msg)

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        return [HeritageUpdateMessage]
