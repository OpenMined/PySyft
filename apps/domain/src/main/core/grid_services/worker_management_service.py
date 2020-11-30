# third party
from nacl.signing import VerifyKey

# syft relative
from syft.core.common.message import ImmediateSyftMessageWithoutReply
from syft.core.node.abstract.node import AbstractNode
from syft.core.node.common.service.auth import service_auth
from syft.core.node.common.service.node_service import ImmediateNodeServiceWithReply
from syft.core.node.domain.service.worker_management_message import CreateWorkerMessage
from syft.core.node.domain.service.worker_management_message import WorkerCreatedMessage
from syft.core.node.vm.client import VirtualMachineClient
from syft.decorators.syft_decorator_impl import syft_decorator
from syft.proto.core.node.common.service.repr_service_pb2 import (
    CreateWorkerMessage as CreateWorkerMessage_PB,
)
from syft.grid.grid_client import proxy_client, connect
from syft.grid.connections.http_connection import HTTPConnection
from syft.core.node.device.client import DeviceClient

class CreateWorkerService(ImmediateNodeServiceWithReply):
    @staticmethod
    def process(
        node: AbstractNode, msg: CreateWorkerMessage, verify_key: VerifyKey
    ) -> None:
        # 1 - Spin up a new VM
        #  -> Return url
        # 2 - Connect with the new VM
        # 3 - Save the client in ->  node._in_memory_client_registry[client.address.vm_id] = client
        #response_msg = node.private_device.create_vm(settings=msg.settings)
        #vm_client = proxy_client(
        #    client=node.private_device,
        #    address=response_msg.vm_address,
        #    client_type=VirtualMachineClient,
        #    proxy_mode=False,
        #)
        #node.in_memory_client_registry[response_msg.vm_address.vm_id] = vm_client
        #return WorkerCreatedMessage(
        #    address=msg.reply_to, vm_address=response_msg.vm_address
        #)

    @staticmethod
    def message_handler_types() -> List[Type[CreateWorkerMessage]]:
        return [CreateWorkerMessage]
