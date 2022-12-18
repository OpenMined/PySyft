# stdlib
from typing import Any
from typing import Dict as DictType
from typing import List
from typing import Optional

# relative
from .....common import UID
from .....common.message import ImmediateSyftMessageWithReply
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.serializable import serializable
from .....io.address import Address


@serializable(recursive_serde=True)
class UpdateRequestHandlerMessage(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        handler: DictType[str, Any],
        address: Address,
        msg_id: Optional[UID] = None,
        keep: bool = True,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.handler = handler
        self.keep = keep


@serializable(recursive_serde=True)
class GetAllRequestHandlersMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = ["id", "address", "reply_to"]

    def __init__(
        self, address: Address, reply_to: Address, msg_id: Optional[UID] = None
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)


@serializable(recursive_serde=True)
class GetAllRequestHandlersResponseMessage(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["handlers", "id", "address"]

    def __init__(
        self,
        handlers: List[DictType],
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.handlers = handlers
