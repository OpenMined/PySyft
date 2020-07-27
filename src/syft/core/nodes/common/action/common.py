from ...abstract.node import AbstractNode
from syft.core.message import SyftMessage
from syft.core.message import SyftMessageWithoutReply
from syft.core.message import SyftMessageWithReply


class Action(SyftMessage):
    ""


class ImmediateActionWithoutReply(Action, SyftMessageWithoutReply):
    ""

    def execute_action(self, node: AbstractNode):
        raise NotImplementedError


class EventualActionWithoutReply(Action, SyftMessageWithoutReply):
    ""

    def execute_action(self, node: AbstractNode):
        raise NotImplementedError


class ImmediateActionWithReply(Action, SyftMessageWithReply):
    ""

    def execute_action(self, node: AbstractNode) -> SyftMessageWithoutReply:
        raise NotImplementedError