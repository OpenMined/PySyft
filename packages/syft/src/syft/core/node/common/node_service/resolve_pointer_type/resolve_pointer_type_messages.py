# stdlib
from typing import Optional

# relative
from .....common.message import ImmediateSyftMessageWithReply
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.serializable import serializable
from .....common.uid import UID


@serializable(recursive_serde=True)
class ResolvePointerTypeMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = ["id_at_location", "address", "reply_to", "id"]

    def __init__(
        self,
        id_at_location: UID,
        address: UID,
        reply_to: UID,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.id_at_location = id_at_location


@serializable(recursive_serde=True)
class ResolvePointerTypeAnswerMessage(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["id", "address", "type_path"]

    def __init__(
        self,
        type_path: str,
        address: UID,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.type_path = type_path
