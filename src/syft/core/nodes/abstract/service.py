from __future__ import annotations
from ....decorators import syft_decorator
from ...message import SyftMessage
from ....common import AbstractNode
from typing import List


class NodeService:
    @staticmethod
    @syft_decorator(typechecking=True)
    def process(node: AbstractNode, msg: SyftMessage) -> object:
        raise NotImplementedError

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        raise NotImplementedError
