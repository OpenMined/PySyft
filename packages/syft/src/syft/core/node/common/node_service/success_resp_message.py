# stdlib
from typing import Optional

# third party
from typing_extensions import final

# relative
from ....common.message import ImmediateSyftMessageWithoutReply
from ....common.serde.serializable import serializable
from ....common.uid import UID
from ....io.address import Address


@serializable(recursive_serde=True)
@final
class SuccessResponseMessage(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["id", "address", "resp_msg"]

    def __init__(
        self,
        address: Address,
        resp_msg: str,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.resp_msg = resp_msg


@serializable(recursive_serde=True)
class ErrorResponseMessage(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["id", "address", "resp_msg"]

    def __init__(
        self,
        address: Address,
        resp_msg: str,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.resp_msg = resp_msg
