# stdlib
from typing import Dict
from typing import Optional

# third party
from typing_extensions import final

# relative
from .....common.message import ImmediateSyftMessageWithReply
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.serializable import serializable
from .....common.uid import UID
from .....io.address import Address


@serializable(recursive_serde=True)
@final
class NetworkSearchMessage(ImmediateSyftMessageWithReply):
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
class NetworkSearchResponse(ImmediateSyftMessageWithoutReply):
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
