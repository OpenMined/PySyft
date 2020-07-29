from ...abstract.node import AbstractNode
from syft.core.message import SyftMessage
from ....message import ImmediateSyftMessageWithoutReply
from ....message import EventualSyftMessageWithoutReply
from ....message import ImmediateSyftMessageWithReply


class Action(SyftMessage):
    ""


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