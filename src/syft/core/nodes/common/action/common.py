from ...abstract.node import AbstractNode
from ....message.syft_message import SyftMessage
from ....message.syft_message import SyftMessageWithoutReply
from ....message.syft_message import SyftMessageWithReply


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