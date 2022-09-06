# DOs and Don's of this class:
# - Do NOT use absolute syft imports (i.e. import syft.core...) Use relative ones.
# - Do NOT put multiple imports on the same line (i.e. from <x> import a, b, c). Use separate lines
# - Do sort imports by length
# - Do group imports by where they come from

# stdlib
from typing import Optional

# relative
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.serializable import serializable
from .....common.uid import UID
from .....io.address import Address


@serializable(recursive_serde=True)
class AcceptOrDenyRequestMessage(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ("id", "address", "accept", "request_id")

    def __init__(
        self,
        accept: bool,
        request_id: UID,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)

        # if false, deny the request
        self.accept = accept
        self.request_id = request_id
