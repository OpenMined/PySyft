from __future__ import annotations
from ..message.repr_message import ReprMessage
from ..message.repr_message import ReprReplyMessage
from ..... import type_hints
from ...abstract.service import NodeService
from .....common import AbstractNode
from typing import List


class ReprService(NodeService):
    @staticmethod
    @type_hints
    def process(node: AbstractNode, msg: ReprMessage) -> ReprReplyMessage:
        return ReprReplyMessage(value=node.__repr__(), route=None)

    @staticmethod
    @type_hints
    def message_handler_types() -> List[type]:
        return [ReprMessage]
