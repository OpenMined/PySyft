from .syft_message import SyftMessage
from ...typecheck import type_hints


class ReprMessage(SyftMessage):
    def __init__(self, route, msg_id=None):
        super().__init__(route=route, msg_id=msg_id)


class ReprReplyMessage(SyftMessage):
    def __init__(self, value, route, msg_id=None):
        super().__init__(route=route, msg_id=msg_id)
        self.value = value

    @type_hints
    def __repr__(self) -> str:
        return f"ReprReplyMessage:{self.value}"
