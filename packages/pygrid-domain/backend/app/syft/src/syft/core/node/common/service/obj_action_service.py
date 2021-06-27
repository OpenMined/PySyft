# stdlib
from typing import List
from typing import Optional
from typing import Type

# third party
from nacl.signing import VerifyKey

# syft relative
from ....common.message import ImmediateSyftMessageWithoutReply
from ...abstract.node import AbstractNode
from ...common.service.node_service import EventualNodeServiceWithoutReply
from ...common.service.node_service import ImmediateNodeServiceWithReply
from ...common.service.node_service import ImmediateNodeServiceWithoutReply
from ..action.common import EventualActionWithoutReply
from ..action.common import ImmediateActionWithReply
from ..action.common import ImmediateActionWithoutReply


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
