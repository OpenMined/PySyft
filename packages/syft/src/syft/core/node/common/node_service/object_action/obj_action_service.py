# stdlib
from typing import List
from typing import Optional
from typing import Type

# third party
from nacl.signing import VerifyKey

# relative
from .....common.message import ImmediateSyftMessageWithoutReply
from ....abstract.node import AbstractNode
from ...action.common import EventualActionWithoutReply
from ...action.common import ImmediateActionWithReply
from ...action.common import ImmediateActionWithoutReply
from ..node_service import EventualNodeServiceWithoutReply
from ..node_service import ImmediateNodeServiceWithReply
from ..node_service import ImmediateNodeServiceWithoutReply


class ImmediateObjectActionServiceWithoutReply(ImmediateNodeServiceWithoutReply):
    @staticmethod
    def process(
        node: AbstractNode,
        msg: ImmediateActionWithoutReply,
        verify_key: Optional[VerifyKey] = None,
    ) -> None:
        msg.execute_action(node=node, verify_key=verify_key)

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateActionWithoutReply]]:
        return [ImmediateActionWithoutReply]


class EventualObjectActionServiceWithoutReply(EventualNodeServiceWithoutReply):
    @staticmethod
    def process(
        node: AbstractNode,
        msg: EventualActionWithoutReply,
        verify_key: Optional[VerifyKey] = None,
    ) -> None:
        msg.execute_action(node=node, verify_key=verify_key)

    @staticmethod
    def message_handler_types() -> List[Type[EventualActionWithoutReply]]:
        return [EventualActionWithoutReply]


class ImmediateObjectActionServiceWithReply(ImmediateNodeServiceWithReply):
    @staticmethod
    def process(
        node: AbstractNode,
        msg: ImmediateActionWithReply,
        verify_key: Optional[VerifyKey] = None,
    ) -> ImmediateSyftMessageWithoutReply:
        return msg.execute_action(node=node, verify_key=verify_key)

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateActionWithReply]]:
        return [ImmediateActionWithReply]
