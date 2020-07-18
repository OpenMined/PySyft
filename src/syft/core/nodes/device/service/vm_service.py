from __future__ import annotations

from ..... import type_hints
from ...abstract.service import WorkerService
from ....message.abstract.vm import AbstractVirtualMachineMessage
from ...common.device import AbstractDevice
from typing import List

class VirtualMachineService(WorkerService):
    @staticmethod
    @type_hints
    def process(worker: AbstractDevice, msg: AbstractVirtualMachineMessage) -> AbstractVirtualMachineMessage:
        return worker.get_vm(name=msg.route.pri_route.vm).send_msg(msg=msg)

    @staticmethod
    @type_hints
    def message_handler_types() -> List[type]:
        return [AbstractVirtualMachineMessage]
