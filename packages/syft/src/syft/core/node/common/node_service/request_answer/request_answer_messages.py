# stdlib
from typing import Callable
from typing import Dict
from typing import Sequence

# relative
from .....common import UID
from .....common.message import ImmediateSyftMessageWithReply
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.serializable import serializable
from .....io.address import Address
from ..request_receiver.request_receiver_messages import RequestStatus


@serializable(recursive_serde=True)
class RequestAnswerMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = ["request_id", "address", "reply_to"]
    __slots__ = ["request_id"]

    def __init__(self, request_id: UID, reply_to: Address, address: Address):
        super().__init__(reply_to, address)
        self.request_id = request_id


@serializable(recursive_serde=True)
class RequestAnswerResponse(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["status", "address", "request_id"]
    __slots__ = ["status", "request_id"]
    __serde_overrides__: Dict[str, Sequence[Callable]] = {
        "status": (
            lambda status: int(status.value),
            lambda int_status: RequestStatus(int(int_status)),
        )
    }

    def __init__(self, status: RequestStatus, request_id: UID, address: Address):
        super().__init__(address)
        self.status = status
        self.request_id = request_id
