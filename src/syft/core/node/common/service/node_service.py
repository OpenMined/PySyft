# stdlib
from typing import Any
from typing import List
from typing import Type

# third party
from nacl.signing import VerifyKey

# syft relative
from .....decorators import syft_decorator
from ....common.message import EventualSyftMessageWithoutReply
from ....common.message import ImmediateSyftMessageWithReply
from ....common.message import ImmediateSyftMessageWithoutReply
from ....common.message import SignedMessageT
from ...abstract.node import AbstractNode


class NodeService:
    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[Type[Any]]:
        raise NotImplementedError

    @classmethod
    def class_name(cls) -> str:
        return str(cls.__name__)

    @classmethod
    def icon(cls) -> str:
        return "⚙️ "

    @classmethod
    def pprint(cls) -> str:
        return f"{cls.icon()} ({cls.class_name()})"


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


class SignedNodeServiceWithoutReply(ImmediateNodeService):
    @staticmethod
    @syft_decorator(typechecking=True)
    def process(node: AbstractNode, msg: SignedMessageT, verify_key: VerifyKey) -> None:
        raise NotImplementedError

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[Type[Any]]:
        raise NotImplementedError
