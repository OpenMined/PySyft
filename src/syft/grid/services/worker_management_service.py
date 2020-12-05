# stdlib
from typing import List
from typing import Type

# third party
from nacl.signing import VerifyKey

# syft relative
from ...core.node.abstract.node import AbstractNode
from ...core.node.common.service.node_service import ImmediateNodeServiceWithReply
from ...core.node.domain.service.worker_management_message import CreateWorkerMessage
from ...core.node.domain.service.worker_management_message import WorkerCreatedMessage


class CreateWorkerService(ImmediateNodeServiceWithReply):
    @staticmethod
    def process(
        node: AbstractNode, msg: CreateWorkerMessage, verify_key: VerifyKey
    ) -> WorkerCreatedMessage:
        # 1 - Spin up a new VM
        #  -> Return url
        # 2 - Connect with the new VM
        # 3 - Save the client in ->  node._in_memory_client_registry[client.address.vm_id] = client
        response_msg = node.private_device.create_vm(settings=msg.settings)  # type: ignore
        return WorkerCreatedMessage(
            address=msg.reply_to, vm_address=response_msg.vm_address
        )

    @staticmethod
    def message_handler_types() -> List[Type[CreateWorkerMessage]]:
        return [CreateWorkerMessage]
