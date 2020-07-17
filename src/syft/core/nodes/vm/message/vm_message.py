from ....message.syft_message import SyftMessage
from .....typecheck import type_hints
from ....message.abstract.vm import AbstractVirtualMachineMessage
from typing import final


@final
class VirtualMachineMessage(SyftMessage, AbstractVirtualMachineMessage):
    def __init__(self, route, msg_id=None):
        super().__init__(route=route, msg_id=msg_id)


@final
class VirtualMachineReplyMessage(SyftMessage, AbstractVirtualMachineMessage):
    def __init__(self, value, route, msg_id=None):
        super().__init__(route=route, msg_id=msg_id)
        self.value = value

    @type_hints
    def __repr__(self) -> str:
        return f"VirtualMachineReplyMessage:{self.value}"
