from __future__ import annotations
from syft.decorators import syft_decorator
from ....message.syft_message import SyftMessage
from ....message.syft_message import SyftMessageWithoutReply
from ...common.node import AbstractNode
from typing import List


class NodeService:
    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        raise NotImplementedError


class NodeServiceWithReply(NodeService):
    @staticmethod
    @syft_decorator(typechecking=True)
    def process(node: AbstractNode, msg: SyftMessage) -> SyftMessageWithoutReply:
        raise NotImplementedError

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        raise NotImplementedError


class NodeServiceWithoutReply(NodeService):
    @staticmethod
    @syft_decorator(typechecking=True)
    def process(node: AbstractNode, msg: SyftMessage) -> SyftMessageWithoutReply:
        raise NotImplementedError

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        raise NotImplementedError
