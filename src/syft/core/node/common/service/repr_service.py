from __future__ import annotations

from typing import List, final

from syft import type_hints
from syft.core.common.message import ImmediateSyftMessageWithoutReply
from syft.core.common.uid import UID

from ....io.address import Address
from ...abstract.node import AbstractNode
from .node_service import ImmediateNodeServiceWithoutReply


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
