from __future__ import annotations

from .....decorators import syft_decorator
from ...abstract.service import WorkerService
from ....message.abstract.vm import AbstractVirtualMachineMessage
from ...common.device import AbstractDevice
from typing import List


class VirtualMachineMessageService(WorkerService):
    @staticmethod
    @syft_decorator(typechecking=True)
    def process(
        worker: AbstractDevice, msg: AbstractVirtualMachineMessage
    ) -> AbstractVirtualMachineMessage:
        return worker.get_vm(id_or_name=msg.route.pri_route.vm).send_msg(msg=msg)

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        return [AbstractVirtualMachineMessage]
