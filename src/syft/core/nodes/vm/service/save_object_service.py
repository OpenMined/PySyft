from __future__ import annotations

from ...abstract.service import NodeService
from ....message import SaveObjectMessage

from ..... import type_hints

from .....common import AbstractNode
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
