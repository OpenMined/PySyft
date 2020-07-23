from __future__ import annotations

from .....decorators import syft_decorator
from ...abstract.service import NodeService
from ..message.lifecycles_messages import VirtualMachineLifecycleMessage
from ..message.lifecycles_messages import CreateVirtualMachineMessage
from ..message.lifecycles_messages import CreateVirtualMachineReplyMessage
from ...common.device import AbstractDevice
from ...vm.vm import VirtualMachine
from typing import List


class VirtualMachineLifecycleService(NodeService):

    def __init__(self, *args, **kwargs):
        self.msg_type_router = {}
        self.msg_type_router[CreateVirtualMachineMessage] = self.create_vm

    @syft_decorator(typechecking=True)
    def process(self, node: AbstractDevice, msg: VirtualMachineLifecycleMessage
    ) -> VirtualMachineLifecycleMessage:
        return self.msg_type_router[type(msg)](node=node, msg=msg)

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        return [VirtualMachineLifecycleMessage]

    def create_vm(self, node:AbstractDevice, msg: CreateVirtualMachineMessage) -> CreateVirtualMachineReplyMessage:
        vm = VirtualMachine(name=msg.vm_name)

        client = vm.get_client()
        node.vms[vm.id] = client
        node.vm_name2id[vm.name] = vm.id

        # route=None because the message is just going back to the Device (i think)
        return CreateVirtualMachineReplyMessage(client=client, route=None)