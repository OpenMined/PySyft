from __future__ import annotations
from .....decorators import type_hints
from ...abstract.service.node_service import NodeService
from ...common.node import AbstractNode
from ....message import DeleteObjectMessage

from typing import List


class DeleteObjectService(NodeService):
    @staticmethod
    @type_hints
    def process(node: AbstractNode, msg: DeleteObjectMessage) -> None:
        return node.store.delete_object(msg.id)

    @staticmethod
    @type_hints
    def message_handler_types() -> List[type]:
        return [DeleteObjectMessage]
