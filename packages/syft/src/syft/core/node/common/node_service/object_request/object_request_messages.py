# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

# third party
from typing_extensions import final

# relative
from .....common.message import ImmediateSyftMessageWithReply
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.serializable import serializable
from .....common.uid import UID
from .....io.address import Address
from ..request_receiver.request_receiver_messages import RequestMessage


@serializable(recursive_serde=True)
@final
class CreateRequestMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = ["address", "id", "reply_to", "content"]

    def __init__(
        self,
        address: Address,
        content: Dict,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.content = content


@serializable(recursive_serde=True)
@final
class CreateBudgetRequestMessage(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["address", "id", "budget", "reason"]

    def __init__(
        self,
        address: Address,
        budget: float,
        reason: str,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.budget = budget
        self.reason = reason


@serializable(recursive_serde=True)
@final
class CreateRequestResponse(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["address", "id", "status_code", "content"]

    def __init__(
        self,
        address: Address,
        status_code: int,
        content: Dict,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.status_code = status_code
        self.content = content


@serializable(recursive_serde=True)
@final
class GetRequestMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = ["address", "id", "reply_to", "request_id"]

    def __init__(
        self,
        address: Address,
        request_id: str,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.request_id = request_id


@serializable(recursive_serde=True)
@final
class GetRequestResponse(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["address", "id", "status_code", "request_id"]

    def __init__(
        self,
        address: Address,
        status_code: int,
        request_id: Dict[str, Any],
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.status_code = status_code
        self.request_id = request_id


@serializable(recursive_serde=True)
@final
class GetRequestsMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = ["address", "id", "reply_to"]

    def __init__(
        self,
        address: Address,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)


@serializable(recursive_serde=True)
@final
class GetRequestsResponse(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["address", "id", "status_code", "content"]

    def __init__(
        self,
        address: Address,
        status_code: int,
        content: List[Dict],
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.status_code = status_code
        self.content = content


@serializable(recursive_serde=True)
@final
class GetBudgetRequestsMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = ["address", "id", "status_code", "reply_to"]

    def __init__(
        self,
        address: Address,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)


@serializable(recursive_serde=True)
@final
class GetBudgetRequestsResponse(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["address", "id", "content"]

    def __init__(
        self,
        address: Address,
        content: List[Dict],
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.content = content


@serializable(recursive_serde=True)
@final
class UpdateRequestMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = ["address", "id", "reply_to", "request_id", "status"]

    def __init__(
        self,
        address: Address,
        request_id: str,
        status: str,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.request_id = request_id
        self.status = status


@serializable(recursive_serde=True)
@final
class UpdateRequestResponse(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["address", "id", "status_code", "request_id", "status"]

    def __init__(
        self,
        address: Address,
        status_code: int,
        status: str,
        request_id: str,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.status_code = status_code
        self.status = status
        self.request_id = request_id


@serializable(recursive_serde=True)
@final
class DeleteRequestMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = ["address", "id", "reply_to", "request_id"]

    def __init__(
        self,
        address: Address,
        request_id: str,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.request_id = request_id


@serializable(recursive_serde=True)
@final
class DeleteRequestResponse(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["address", "id", "status_code", "request_id"]

    def __init__(
        self,
        address: Address,
        status_code: int,
        request_id: str,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.status_code = status_code
        self.request_id = request_id


@serializable(recursive_serde=True)
class GetAllRequestsMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = ["address", "id", "reply_to"]

    def __init__(
        self, address: Address, reply_to: Address, msg_id: Optional[UID] = None
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)


@serializable(recursive_serde=True)
class GetAllRequestsResponseMessage(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["address", "id", "requests"]

    def __init__(
        self,
        requests: List[RequestMessage],
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.requests = requests
