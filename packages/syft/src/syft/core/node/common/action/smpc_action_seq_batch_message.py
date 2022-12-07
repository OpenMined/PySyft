# future
from __future__ import annotations

# stdlib
from typing import List
from typing import Optional

# relative
from ....common.message import ImmediateSyftMessageWithoutReply
from ....common.serde.serializable import serializable
from ....common.uid import UID
from ....io.address import Address
from .smpc_action_message import SMPCActionMessage


@serializable(recursive_serde=True)
class SMPCActionSeqBatchMessage(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["smpc_actions", "address", "id"]

    def __init__(
        self,
        smpc_actions: List[SMPCActionMessage],
        address: Address,
        msg_id: Optional[UID] = None,
    ) -> None:
        self.smpc_actions = smpc_actions
        super().__init__(address=address, msg_id=msg_id)

    def __str__(self) -> str:
        res = "SMPCActionSeqBatch:\n"
        res = f"{res} {self.smpc_actions}"
        return res

    __repr__ = __str__
