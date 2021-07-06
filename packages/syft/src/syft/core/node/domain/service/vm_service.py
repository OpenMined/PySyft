# stdlib
from queue import Queue
from threading import Thread
import time
from typing import List
from typing import Optional

# third party
from nacl.signing import VerifyKey

# syft relative
from .....util import traceback_and_raise
from ...abstract.node import AbstractNode
from ...common.action.smpc_action import SMPCAction
from ...common.service.node_service import ImmediateNodeServiceWithReply
from ...common.service.node_service import ImmediateNodeServiceWithoutReply
from .request_answer_message import RequestAnswerMessage
from .request_answer_message import RequestAnswerResponse
from .request_message import RequestMessage
from .request_message import RequestStatus


class VMRequestService(ImmediateNodeServiceWithoutReply):
    @staticmethod
    def message_handler_types() -> List[type]:
        return [RequestMessage]

    @staticmethod
    def process(
        node: AbstractNode, msg: RequestMessage, verify_key: Optional[VerifyKey] = None
    ) -> None:
        """ """


class VMRequestAnswerMessageService(ImmediateNodeServiceWithReply):
    @staticmethod
    def message_handler_types() -> List[type]:
        return [RequestAnswerMessage]

    @staticmethod
    def process(
        node: AbstractNode,
        msg: RequestAnswerMessage,
        verify_key: Optional[VerifyKey] = None,
    ) -> RequestAnswerResponse:
        if verify_key is None:
            traceback_and_raise(
                ValueError(
                    "Can't process Request service without a given " "verification key"
                )
            )

        status = RequestStatus.Rejected
        address = msg.reply_to
        if node.root_verify_key == verify_key or node.vm_id == address.vm_id:
            status = RequestStatus.Accepted

        return RequestAnswerResponse(
            request_id=msg.request_id, address=address, status=status
        )


def consume_smpc_actions_round_robin(queue):
    # Queue keeps a list of actions

    max_nr_retries = 10
    last_msg_id = None
    while True:
        element = queue.get()
        node, msg, verify_key, nr_retries = element

        print("Executing Actions")
        if nr_retries > max_nr_retries:
            raise ValueError(f"Retries to many times for {element}")

        if last_msg_id is not None and last_msg_id == msg.id:
            # If there is only one list of actiosn
            time.sleep(1)

        last_msg_id = msg.id
        try:
            msg.execute_action(node, verify_key)
        except KeyError:
            queue.put((node, msg, verify_key, nr_retries + 1))


queue = Queue()
thread_smpc_action = Thread(
    target=consume_smpc_actions_round_robin, args=(queue,), daemon=True
)
thread_smpc_action.start()


class VMSMPCService(ImmediateNodeServiceWithoutReply):
    @staticmethod
    def message_handler_types() -> List[type]:
        return [SMPCAction]

    @staticmethod
    def process(
        node: AbstractNode, msg: RequestMessage, verify_key: Optional[VerifyKey] = None
    ) -> None:
        queue.put((node, msg, verify_key, 0))
