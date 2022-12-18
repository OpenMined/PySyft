# stdlib
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
class GetRemainingBudgetMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = ["id", "address", "reply_to"]

    def __init__(
        self,
        address: Address,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)


@serializable(recursive_serde=True)
class GetRemainingBudgetReplyMessage(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["id", "address", "budget"]

    def __init__(
        self,
        budget: float,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.budget = budget
