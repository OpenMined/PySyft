from __future__ import annotations

from .....decorators import syft_decorator
from ...common.service.node_service import NodeServiceWithoutReply
from ...common.service.node_service import NodeServiceWithReply
from ...abstract.node import AbstractNode
from ....message.syft_message import SyftMessageWithReply
from ..action.common import ActionWithReply
from ..action.common import ActionWithoutReply
from typing import List


class ObjectActionServiceWithoutReply(NodeServiceWithoutReply):

    @syft_decorator(typechecking=True)
    def process(self, node: AbstractNode, msg: ActionWithoutReply) -> None:
        msg.execute_action(node=node)

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        return [ActionWithoutReply]

class ObjectActionServiceWithReply(NodeServiceWithReply):

    @syft_decorator(typechecking=True)
    def process(self, node: AbstractNode, msg: ActionWithoutReply) -> SyftMessageWithReply:
        return msg.execute_action(node=node)

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        return [ActionWithReply]
