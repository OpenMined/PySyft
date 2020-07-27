from __future__ import annotations
from syft.decorators import syft_decorator
from syft.core.message import SyftMessage
from syft.core.message import ImmediateSyftMessageWithoutReply
from ...abstract.node import AbstractNode
from typing import List


class NodeService:
    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        raise NotImplementedError

class ImmediateNodeService(NodeService):
    """A service for messages which should be immediately executed"""


class EventualNodeService(NodeService):
    """A service for messages which need not be immediately executed
    but which can be executed at the worker's convenience"""


class ImmediateNodeServiceWithReply(ImmediateNodeService):
    @staticmethod
    @syft_decorator(typechecking=True)
    def process(node: AbstractNode, msg: SyftMessage) -> ImmediateSyftMessageWithoutReply:
        raise NotImplementedError

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        raise NotImplementedError


class ImmediateNodeServiceWithoutReply(NodeService):
    @staticmethod
    @syft_decorator(typechecking=True)
    def process(node: AbstractNode, msg: SyftMessage) -> ImmediateSyftMessageWithoutReply:
        raise NotImplementedError

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        raise NotImplementedError
