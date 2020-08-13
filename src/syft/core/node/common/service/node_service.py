# external class imports
from nacl.signing import VerifyKey
from typing import List
from typing import Type
from typing import Any

from syft.core.common.message import (
    EventualSyftMessageWithoutReply,
    ImmediateSyftMessageWithoutReply,
    ImmediateSyftMessageWithReply,
    SignedMessageT,
)
from syft.decorators import syft_decorator

from ...abstract.node import AbstractNode


class NodeService:
    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[Type[Any]]:
        raise NotImplementedError


class ImmediateNodeService(NodeService):
    """A service for messages which should be immediately executed"""

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[Type[Any]]:
        raise NotImplementedError


class EventualNodeService(NodeService):
    """A service for messages which need not be immediately executed
    but which can be executed at the worker's convenience"""

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[Type[Any]]:
        raise NotImplementedError


class ImmediateNodeServiceWithReply(ImmediateNodeService):
    @staticmethod
    @syft_decorator(typechecking=True)
    def process(
        node: AbstractNode, msg: ImmediateSyftMessageWithReply, verify_key: VerifyKey
    ) -> ImmediateSyftMessageWithoutReply:
        raise NotImplementedError

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[Type[Any]]:
        raise NotImplementedError


class EventualNodeServiceWithoutReply(NodeService):
    @staticmethod
    @syft_decorator(typechecking=True)
    def process(
        node: AbstractNode, msg: EventualSyftMessageWithoutReply, verify_key: VerifyKey
    ) -> None:
        raise NotImplementedError

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[Type[Any]]:
        raise NotImplementedError


class ImmediateNodeServiceWithoutReply(NodeService):
    @staticmethod
    @syft_decorator(typechecking=True)
    def process(
        node: AbstractNode, msg: ImmediateSyftMessageWithoutReply, verify_key: VerifyKey
    ) -> None:
        raise NotImplementedError

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[Type[Any]]:
        raise NotImplementedError


class SignedNodeServiceWithReply(ImmediateNodeService):
    @staticmethod
    @syft_decorator(typechecking=True)
    def process(
        node: AbstractNode, msg: SignedMessageT, verify_key: VerifyKey
    ) -> SignedMessageT:
        raise NotImplementedError

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[Type[Any]]:
        raise NotImplementedError
