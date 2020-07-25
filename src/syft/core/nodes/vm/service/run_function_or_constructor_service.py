from __future__ import annotations
from ...abstract.service.node_service import NodeService
from ...common.node import AbstractNode
from ....message import RunFunctionOrConstructorMessage
from ..... import type_hints
from typing import List


class RunFunctionOrConstructorService(NodeService):
    @staticmethod
    @type_hints
    def process(node: AbstractNode, msg: RunFunctionOrConstructorMessage) -> None:
        pass

    @staticmethod
    @type_hints
    def message_handler_types() -> List[type]:
        return [RunFunctionOrConstructorMessage]
