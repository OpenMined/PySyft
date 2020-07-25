"""The purpose of this service is to inform lower level devices
of changes in the hierarchy above them. For example, if a Domain
registers within a new Network or if a Device registers within
a new Domain, all the other child nodes will need to know this
information to populate complete addresses into their clients."""


from __future__ import annotations

from .....decorators import syft_decorator
from .node_service import NodeService
from ...common.node import AbstractNode
from ....io.address import Address
from .....common.id import UID
from ....message.syft_message import SyftMessageWithoutReply
from typing import List

class HeritageUpdateMessage(SyftMessageWithoutReply):

    def __init__(self, id_type:str, new_ancestor_id: UID, address: Address, msg_id: UID = None):
        super().__init__(address=address, msg_id=msg_id)
        self.id_type = id_type
        self.new_ancestor_id = new_ancestor_id

class HeritageUpdateService(NodeService):

    @syft_decorator(typechecking=True)
    def process(self, node: AbstractNode, msg: HeritageUpdateMessage) -> None:
        print(f"Updating {msg.id_type} to {msg.new_ancestor_id} on note {node}")
        node.__setattr__(msg.id_type, msg.new_ancestor_id)
        return msg

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        return [HeritageUpdateMessage]