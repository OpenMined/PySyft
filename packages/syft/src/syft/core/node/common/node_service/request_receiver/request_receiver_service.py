# stdlib
import time
from typing import List
from typing import Optional

# third party
from nacl.signing import VerifyKey

# relative
from ......logger import traceback_and_raise
from ....abstract.node import AbstractNode
from ...node import DuplicateRequestException
from ..node_service import ImmediateNodeServiceWithoutReply
from .request_receiver_messages import RequestMessage


class RequestReceiverService(ImmediateNodeServiceWithoutReply):
    @staticmethod
    def message_handler_types() -> List[type]:
        return [RequestMessage]

    @staticmethod
    def process(
        node: AbstractNode, msg: RequestMessage, verify_key: Optional[VerifyKey] = None
    ) -> None:
        if verify_key is None:
            traceback_and_raise(
                ValueError(
                    "Can't process Request service without a given " "verification key"
                )
            )

        if msg.requester_verify_key != verify_key:
            traceback_and_raise(
                Exception(
                    "You tried to request access for a key that is not yours!"
                    "You cannot do this! Whatever key you want to request access"
                    "for must be the verify key that also verifies the message"
                    "containing the request."
                )
            )

        # since we reject/accept requests based on the ID, we don't want there to be
        # multiple requests with the same ID because this could cause security problems.
        for req in node.requests:
            # the same user has requested the same object so we raise a
            # DuplicateRequestException
            if (
                req.object_id == msg.object_id
                and req.requester_verify_key == msg.requester_verify_key
            ):
                traceback_and_raise(
                    DuplicateRequestException(
                        f"You have already requested {msg.object_id}"
                    )
                )

        # using the local arrival time we can expire the request
        msg.set_arrival_time(arrival_time=time.time())

        # At receiving a request from DS, we clear it's object_tags, and re-set it as the
        # tags of the requested object. Because the DS may give fake tags.
        while msg.object_tags:
            msg.object_tags.pop()

        if "budget" not in msg.object_type:
            msg.object_tags.extend(node.store.get(msg.object_id, proxy_only=True)._tags)

        node.requests.append(msg)
