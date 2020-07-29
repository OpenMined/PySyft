from __future__ import annotations

from .....decorators import syft_decorator
from ...common.service.node_service import ImmediateNodeServiceWithoutReply
from ...common.service.node_service import ImmediateNodeServiceWithReply
from ...common.service.node_service import EventualNodeServiceWithoutReply
from ...abstract.node import AbstractNode
from syft.core.message import ImmediateSyftMessageWithoutReply
from ..action.common import ImmediateActionWithReply
from ..action.common import ImmediateActionWithoutReply
from ..action.common import EventualActionWithoutReply
from typing import List


class ObjectActionServiceWithoutReply(ImmediateNodeServiceWithoutReply):
    @syft_decorator(typechecking=True)
    def process(self, node: AbstractNode, msg: ImmediateActionWithoutReply) -> None:
        msg.execute_action(node=node)

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        return [ImmediateActionWithoutReply]


class ObjectActionServiceWithReply(ImmediateNodeServiceWithReply):
    @syft_decorator(typechecking=True)
    def process(
        self, node: AbstractNode, msg: ImmediateActionWithReply
    ) -> ImmediateSyftMessageWithoutReply:
        return msg.execute_action(node=node)

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        return [ImmediateActionWithReply]
