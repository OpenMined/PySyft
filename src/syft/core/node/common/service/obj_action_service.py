from typing import List

from syft.core.common.message import ImmediateSyftMessageWithoutReply

from .....decorators import syft_decorator
from ...abstract.node import AbstractNode
from ...common.service.node_service import (
    EventualNodeServiceWithoutReply,
    ImmediateNodeServiceWithoutReply,
    ImmediateNodeServiceWithReply,
)
from ..action.common import (
    EventualActionWithoutReply,
    ImmediateActionWithoutReply,
    ImmediateActionWithReply,
)


class ImmediateObjectActionServiceWithoutReply(ImmediateNodeServiceWithoutReply):
    @syft_decorator(typechecking=True)
    def process(self, node: AbstractNode, msg: ImmediateActionWithoutReply) -> None:
        msg.execute_action(node=node)

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        return [ImmediateActionWithoutReply]


class EventualObjectActionServiceWithoutReply(EventualNodeServiceWithoutReply):
    @syft_decorator(typechecking=True)
    def process(self, node: AbstractNode, msg: EventualActionWithoutReply) -> None:
        msg.execute_action(node=node)

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        return [EventualActionWithoutReply]


class ImmediateObjectActionServiceWithReply(ImmediateNodeServiceWithReply):
    @syft_decorator(typechecking=True)
    def process(
        self, node: AbstractNode, msg: ImmediateActionWithReply
    ) -> ImmediateSyftMessageWithoutReply:
        return msg.execute_action(node=node)

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        return [ImmediateActionWithReply]
