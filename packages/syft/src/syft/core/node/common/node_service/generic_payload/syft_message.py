# stdlib
from typing import Any
from typing import Dict
from typing import Optional

# third party
from nacl.signing import VerifyKey

# relative
from .....common.message import ImmediateSyftMessage, SignedMessage
from .....common.uid import UID
from .....io.address import Address
from ....abstract.node_service_interface import NodeServiceInterface

class SyftMessage(ImmediateSyftMessage):
    __attr_allowlist__ = ["id", "payload", "address", "reply_to", "msg_id", "kwargs"]

    signed_type = SignedMessage

    def __init__(
        self,
        address: Address,
        kwargs: Dict[str,Any] = {},
        msg_id: Optional[UID] = None,
        reply_to: Optional[Address] = None) -> None:
        super().__init__(address=address, msg_id=msg_id)
        self.reply_to = reply_to
        self.kwargs = kwargs        
    
    def run(self, node: NodeServiceInterface, verify_key: Optional[VerifyKey] = None) -> ImmediateSyftMessage:
        raise NotImplementedError
