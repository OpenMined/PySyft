from syft.core.common.message import (
    EventualSyftMessageWithoutReply,
    ImmediateSyftMessageWithoutReply,
    ImmediateSyftMessageWithReply,
    SyftMessage,
)

from ...abstract.node import AbstractNode


class Action(SyftMessage):
    ""

    def execute_action(self, node: AbstractNode) -> None:
        raise NotImplementedError


class ImmediateActionWithoutReply(Action, ImmediateSyftMessageWithoutReply):
    ""

    def execute_action(self, node: AbstractNode) -> None:
        raise NotImplementedError


class EventualActionWithoutReply(Action, EventualSyftMessageWithoutReply):
    ""

    def execute_action(self, node: AbstractNode) -> None:
        raise NotImplementedError


class ImmediateActionWithReply(Action, ImmediateSyftMessageWithReply):
    ""

    def execute_action(self, node: AbstractNode) -> ImmediateSyftMessageWithoutReply:
        raise NotImplementedError
