from .vm_message import VirtualMachineMessage
from .vm_message import VirtualMachineReplyMessage
from .....typecheck import type_hints
from typing import final


@final
class ReprMessage(VirtualMachineMessage):
    def __init__(self, route, msg_id=None):
        super().__init__(route=route, msg_id=msg_id)


@final
class ReprReplyMessage(VirtualMachineReplyMessage):
    def __init__(self, value, route, msg_id=None):
        super().__init__(route=route, msg_id=msg_id)
        self.value = value

    @type_hints
    def __repr__(self) -> str:
        return f"ReprReplyMessage:{self.value}"
