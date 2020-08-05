from typing import List, Type, Any

from syft.core.common.message import (
    EventualSyftMessageWithoutReply,
    ImmediateSyftMessageWithoutReply,
    ImmediateSyftMessageWithReply,
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
        node: AbstractNode, msg: SyftMessage
    ) -> ImmediateSyftMessageWithoutReply:
        raise NotImplementedError

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[Type[Any]]:
        raise NotImplementedError


class EventualNodeServiceWithoutReply(NodeService):
    @staticmethod
    @syft_decorator(typechecking=True)
    def process(node: AbstractNode, msg: EventualSyftMessageWithoutReply) -> None:

        raise NotImplementedError

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[Type[Any]]:
        raise NotImplementedError


class ImmediateNodeServiceWithoutReply(NodeService):
    @staticmethod
    @syft_decorator(typechecking=True)
    def process(node: AbstractNode, msg: ImmediateSyftMessageWithoutReply) -> None:
        raise NotImplementedError

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[Type[Any]]:
        raise NotImplementedError
