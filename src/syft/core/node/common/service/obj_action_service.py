from typing import List
from typing import Type
from nacl.signing import VerifyKey

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
    def process(self, node: AbstractNode, msg: ImmediateActionWithoutReply, verify_key: VerifyKey) -> None:
        msg.execute_action(node=node, verify_key=verify_key)

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[Type[ImmediateActionWithoutReply]]:
        return [ImmediateActionWithoutReply]


class EventualObjectActionServiceWithoutReply(EventualNodeServiceWithoutReply):
    @syft_decorator(typechecking=True)
    def process(self, node: AbstractNode, msg: EventualActionWithoutReply, verify_key: VerifyKey) -> None:
        msg.execute_action(node=node, verify_key=verify_key)

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[Type[EventualActionWithoutReply]]:
        return [EventualActionWithoutReply]


class ImmediateObjectActionServiceWithReply(ImmediateNodeServiceWithReply):
    @syft_decorator(typechecking=True)
    def process(
        self, node: AbstractNode, msg: ImmediateActionWithReply, verify_key: VerifyKey
    ) -> ImmediateSyftMessageWithoutReply:
        return msg.execute_action(node=node, verify_key=verify_key)

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[Type[ImmediateActionWithReply]]:
        return [ImmediateActionWithReply]
