from __future__ import annotations

from ..... import type_hints
from ....message import GetObjectMessage
from ...abstract.service.node_service import NodeService
from ...common.node import AbstractNode
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
