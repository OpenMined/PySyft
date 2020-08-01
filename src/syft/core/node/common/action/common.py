from ...abstract.node import AbstractNode
from syft.core.common.message import SyftMessage
from syft.core.common.message import ImmediateSyftMessageWithoutReply
from syft.core.common.message import EventualSyftMessageWithoutReply
from syft.core.common.message import ImmediateSyftMessageWithReply


class Action(SyftMessage):
    ""

    def execute_action(self, node: AbstractNode):
        raise NotImplementedError


class ImmediateActionWithoutReply(Action, ImmediateSyftMessageWithoutReply):
    ""

    def execute_action(self, node: AbstractNode):
        raise NotImplementedError


class EventualActionWithoutReply(Action, EventualSyftMessageWithoutReply):
    ""

    def execute_action(self, node: AbstractNode) -> None:
        raise NotImplementedError


class ImmediateActionWithReply(Action, ImmediateSyftMessageWithReply):
    ""

    def execute_action(self, node: AbstractNode) -> ImmediateSyftMessageWithoutReply:
        raise NotImplementedError
