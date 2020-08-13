from typing import Union
from syft.core.common.message import (
    EventualSyftMessageWithoutReply,
    ImmediateSyftMessageWithoutReply,
    ImmediateSyftMessageWithReply,
    SyftMessage,
)
from nacl.signing import VerifyKey
from ...abstract.node import AbstractNode


class Action(SyftMessage):
    ""

    def execute_action(
        self, node: AbstractNode, verify_key: VerifyKey
    ) -> Union[SyftMessage, None]:
        raise NotImplementedError


class ImmediateActionWithoutReply(Action, ImmediateSyftMessageWithoutReply):
    ""

    def execute_action(self, node: AbstractNode, verify_key: VerifyKey) -> None:
        raise NotImplementedError


class EventualActionWithoutReply(Action, EventualSyftMessageWithoutReply):
    ""

    def execute_action(self, node: AbstractNode, verify_key: VerifyKey) -> None:
        raise NotImplementedError


class ImmediateActionWithReply(Action, ImmediateSyftMessageWithReply):
    ""

    def execute_action(
        self, node: AbstractNode, verify_key: VerifyKey
    ) -> ImmediateSyftMessageWithoutReply:
        raise NotImplementedError
