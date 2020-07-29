"""The purpose of this service is to inform lower level devices
of changes in the hierarchy above them. For example, if a Domain
registers within a new Network or if a Device registers within
a new Domain, all the other child nodes will need to know this
information to populate complete addresses into their clients."""


from __future__ import annotations

from .....decorators import syft_decorator
from .node_service import ImmediateNodeServiceWithoutReply
from ...abstract.node import AbstractNode
from ....io.address import Address
from .....common.id import UID
from syft.core.message import ImmediateSyftMessageWithoutReply
from typing import List

#TODO: change all old_message names in syft to have "WithReply" or "WithoutReply" at teh end of the name


class HeritageUpdateMessage(ImmediateSyftMessageWithoutReply):
    def __init__(
        self, new_ancestry_address:Address, address: Address, msg_id: UID = None
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.new_ancestry_address = new_ancestry_address


class HeritageUpdateService(ImmediateNodeServiceWithoutReply):
    @syft_decorator(typechecking=True)
    def process(self, node: AbstractNode, msg: HeritageUpdateMessage) -> None:
        print(f"Updating to {msg.new_ancestry_address} on note {node}")
        addr = msg.new_ancestry_address
        if(addr.pub_address.network is not None):
            node.network_id = addr.pub_address.network
        if(addr.pub_address.domain is not None):
            node.domain_id = addr.pub_address.domain
        if(addr.pri_address.device is not None):
            node.device_id = addr.pri_address.device

        # TODO: solve this with node group address?
        for node_client in node.known_child_nodes:
            msg.address = node_client.address
            node_client.send_immediate_msg_without_reply(msg=msg)


    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        return [HeritageUpdateMessage]
