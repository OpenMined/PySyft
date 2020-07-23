from __future__ import annotations

from ...abstract.service import NodeService
from ....message import RunFunctionOrConstructorMessage
from ..... import type_hints
from .....common import AbstractNode
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
