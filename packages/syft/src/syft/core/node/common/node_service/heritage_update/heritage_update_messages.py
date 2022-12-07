# stdlib
from typing import Optional

# relative
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.serializable import serializable
from .....common.uid import UID
from .....io.address import Address


@serializable(recursive_serde=True)
class HeritageUpdateMessage(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["new_ancestry_address", "address", "id"]

    def __init__(
        self,
        new_ancestry_address: Address,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.new_ancestry_address = new_ancestry_address
