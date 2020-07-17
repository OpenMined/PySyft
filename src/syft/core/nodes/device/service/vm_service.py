from __future__ import annotations

from ..... import type_hints
from ...abstract.service import WorkerService
from ....message.abstract.vm import AbstractVirtualMachineMessage
from ..device import Device
from typing import List

class VMService(WorkerService):
    @staticmethod
    @type_hints
    def process(device: Device, msg: AbstractVirtualMachineMessage) -> AbstractVirtualMachineMessage:
        return device.vms[msg.route.pri_route.vm].recv_msg(msg)

    @staticmethod
    @type_hints
    def message_handler_types() -> List[type]:
        return [AbstractVirtualMachineMessage]
