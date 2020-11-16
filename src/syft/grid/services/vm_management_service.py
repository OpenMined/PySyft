# stdlib
from typing import List
from typing import Type

# third party
from nacl.signing import VerifyKey

# syft relative
from ...core.common.message import ImmediateSyftMessageWithoutReply
from ...core.node.abstract.node import AbstractNode
from ...core.node.common.service.auth import service_auth
from ...core.node.common.service.node_service import ImmediateNodeServiceWithReply
from ...core.node.device.service.vm_management_message import CreateVMMessage
from ...core.node.device.service.vm_management_message import VMCreatedMessage
from ...core.node.vm.vm import VirtualMachine
from ...decorators.syft_decorator_impl import syft_decorator
from ...proto.core.node.common.service.repr_service_pb2 import (
    CreateVMMessage as CreateVMMessage_PB,
)


class CreateVMService(ImmediateNodeServiceWithReply):
    @staticmethod
    def process(
        node: AbstractNode, msg: CreateVMMessage, verify_key: VerifyKey
    ) -> None:
        # 1 - Spin up a new VM
        #  -> Return url
        # 2 - Connect with the new VM
        # 3 - Save the client in ->  node._in_memory_client_registry[client.address.vm_id] = client
        new_vm = VirtualMachine(
            network=node.network, domain=node.domain, device=node.device
        )
        node.in_memory_client_registry[new_vm.vm_id] = new_vm.get_root_client()
        return VMCreatedMessage(address=msg.reply_to, vm_address=new_vm.address)

    @staticmethod
    def message_handler_types() -> List[Type[CreateVMMessage]]:
        return [CreateVMMessage]
