from __future__ import annotations

from .....decorators import syft_decorator
from ...abstract.service import NodeService
from ....message.abstract.vm import AbstractVirtualMachineMessage
from ...common.device import AbstractDevice
from typing import List


class VirtualMachineMessageService(NodeService):
    @staticmethod
    @syft_decorator(typechecking=True)
    def process(
        node: AbstractDevice, msg: AbstractVirtualMachineMessage
    ) -> AbstractVirtualMachineMessage:
        return node.get_vm(id_or_name=msg.route.pri_route.vm).send_msg(msg=msg)

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        return [AbstractVirtualMachineMessage]
