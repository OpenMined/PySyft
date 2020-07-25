from __future__ import annotations


from ....message import SaveObjectMessage

from ..... import type_hints

from ...abstract.service.node_service import NodeService
from ...common.node import AbstractNode
from typing import List


class SaveObjectService(NodeService):
    @staticmethod
    @type_hints
    def process(node: AbstractNode, msg: SaveObjectMessage) -> None:
        node.store.store_object(msg.id, msg.obj)
        pass

    @staticmethod
    @type_hints
    def message_handler_types() -> List[type]:
        return [SaveObjectMessage]
