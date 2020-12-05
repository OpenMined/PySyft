# stdlib
from typing import List
from typing import Type

# third party
from nacl.signing import VerifyKey

# syft relative
from ...core.node.abstract.node import AbstractNode
from ...core.node.common.service.node_service import ImmediateNodeServiceWithReply
from ...core.node.device.service.vm_management_message import CreateVMMessage
from ...core.node.device.service.vm_management_message import CreateVMResponseMessage
from ...core.node.vm.vm import VirtualMachine


class CreateVMService(ImmediateNodeServiceWithReply):
    @staticmethod
    def process(
        node: AbstractNode, msg: CreateVMMessage, verify_key: VerifyKey
    ) -> CreateVMResponseMessage:
        # 1 - Spin up a new VM
        #  -> Return url
        # 2 - Connect with the new VM
        # 3 - Save the client in ->  node._in_memory_client_registry[client.address.vm_id] = client
        new_vm = VirtualMachine(
            network=node.network, domain=node.domain, device=node.device
        )
        node.in_memory_client_registry[new_vm.vm_id] = new_vm.get_root_client()
        return CreateVMResponseMessage(
            address=msg.reply_to,
            success=True,
            msg="Worker initialized successfully!",
            vm_address=new_vm.address,
        )

    @staticmethod
    def message_handler_types() -> List[Type[CreateVMMessage]]:
        return [CreateVMMessage]
