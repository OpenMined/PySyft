# stdlib
from typing import Union

# third party
from nacl.signing import VerifyKey

# relative
from ....common.message import EventualSyftMessageWithoutReply
from ....common.message import ImmediateSyftMessageWithReply
from ....common.message import ImmediateSyftMessageWithoutReply
from ....common.message import SyftMessage
from ...abstract.node import AbstractNode


class Action(SyftMessage):
    """ """

    def execute_action(
        self, node: AbstractNode, verify_key: VerifyKey
    ) -> Union[SyftMessage, None]:
        raise NotImplementedError


class ImmediateActionWithoutReply(Action, ImmediateSyftMessageWithoutReply):
    """ """

    def execute_action(self, node: AbstractNode, verify_key: VerifyKey) -> None:
        raise NotImplementedError


class EventualActionWithoutReply(Action, EventualSyftMessageWithoutReply):
    """ """

    def execute_action(self, node: AbstractNode, verify_key: VerifyKey) -> None:
        raise NotImplementedError


class ImmediateActionWithReply(Action, ImmediateSyftMessageWithReply):
    """ """

    def execute_action(
        self, node: AbstractNode, verify_key: VerifyKey
    ) -> ImmediateSyftMessageWithoutReply:
        raise NotImplementedError
