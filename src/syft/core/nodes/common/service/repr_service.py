from __future__ import annotations
from syft import type_hints
from typing import List
from typing import final
from .....decorators.syft_decorator import syft_decorator

from .node_service import ImmediateNodeServiceWithoutReply
from ...abstract.node import AbstractNode
from ....message.syft_message import ImmediateSyftMessageWithoutReply
from ....io.address import Address
from .....common.id import UID


@final
class ReprMessage(ImmediateSyftMessageWithoutReply):
    def __init__(self, address: Address, msg_id: (UID, None) = None):
        super().__init__(address=address, msg_id=msg_id)


class ReprService(ImmediateNodeServiceWithoutReply):
    @staticmethod
    @type_hints
    def process(node: AbstractNode, msg: ReprMessage) -> None:
        print(node.__repr__())

    @staticmethod
    @type_hints
    def message_handler_types() -> List[type]:
        return [ReprMessage]
