# stdlib
from typing import Optional

# relative
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.serializable import serializable
from .....common.uid import UID
from .....io.address import Address


@serializable(recursive_serde=True)
class RegisterChildNodeMessage(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["lookup_id", "child_node_client_address", "address", "id"]

    def __init__(
        self,
        lookup_id: UID,
        child_node_client_address: Address,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.lookup_id = lookup_id
        self.child_node_client_address = child_node_client_address
