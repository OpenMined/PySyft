from ....message.syft_message import SyftMessage
from .....decorators import type_hints
from ....message.abstract.device import AbstractDeviceMessage
from typing import final


@final
class DeviceMessage(SyftMessage, AbstractDeviceMessage):
    def __init__(self, route, msg_id=None):
        super().__init__(route=route, msg_id=msg_id)

    @type_hints
    def __repr__(self) -> str:
        return f"DeviceMessage:{self.value}"


@final
class DeviceReplyMessage(SyftMessage, AbstractDeviceMessage):
    def __init__(self, route, msg_id=None):
        super().__init__(route=route, msg_id=msg_id)

    @type_hints
    def __repr__(self) -> str:
        return f"DeviceReplyMessage:{self.value}"
