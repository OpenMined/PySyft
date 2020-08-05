from typing import List, Type, Optional
from typing_extensions import final

from syft.decorators import type_hints
from syft.core.common.message import ImmediateSyftMessageWithoutReply
from syft.core.common.uid import UID

from ....io.address import Address
from ...abstract.node import AbstractNode
from .node_service import ImmediateNodeServiceWithoutReply


@final
class ReprMessage(ImmediateSyftMessageWithoutReply):
    def __init__(self, address: Address, msg_id: Optional[UID] = None):
        super().__init__(address=address, msg_id=msg_id)


class ReprService(ImmediateNodeServiceWithoutReply):
    @staticmethod
    @type_hints
    def process(node: AbstractNode, msg: ReprMessage) -> None:
        print(node.__repr__())

    @staticmethod
    @type_hints
    def message_handler_types() -> List[Type[ReprMessage]]:
        return [ReprMessage]
