# stdlib
from typing import Optional

# relative
from .....common.message import ImmediateSyftMessageWithReply
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.serializable import serializable
from .....common.uid import UID
from .....io.address import Address


@serializable(recursive_serde=True)
class GetReprMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = ["id", "address", "id_at_location", "reply_to"]

    def __init__(
        self,
        address: Address,
        reply_to: Address,
        id_at_location: UID,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.id_at_location = id_at_location


@serializable(recursive_serde=True)
class GetReprReplyMessage(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["id", "address", "repr"]

    def __init__(
        self,
        repr: str,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.repr = repr
