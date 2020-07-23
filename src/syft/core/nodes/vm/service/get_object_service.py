from __future__ import annotations

from ..... import type_hints
from ...abstract.service import NodeService
from ....message import GetObjectMessage
from .....common import AbstractNode
from typing import List


class GetObjectService(NodeService):
    @staticmethod
    @type_hints
    def process(
        node: AbstractNode, msg: GetObjectMessage
    ) -> object:  # TODO: return StoreableObject
        return node.store.get_object(msg.id)

    @staticmethod
    @type_hints
    def message_handler_types() -> List[type]:
        return [GetObjectMessage]
