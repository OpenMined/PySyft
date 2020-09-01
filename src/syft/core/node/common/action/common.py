# stdlib
from typing import Union

# third party
from nacl.signing import VerifyKey

# syft absolute
from syft.core.common.message import EventualSyftMessageWithoutReply
from syft.core.common.message import ImmediateSyftMessageWithReply
from syft.core.common.message import ImmediateSyftMessageWithoutReply
from syft.core.common.message import SyftMessage

# syft relative
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
