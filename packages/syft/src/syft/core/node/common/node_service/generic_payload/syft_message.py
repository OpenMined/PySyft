# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

# third party
from nacl.signing import VerifyKey
from pydantic import BaseModel

# relative
from .....common.message import ImmediateSyftMessage
from .....common.message import SignedMessage
from .....common.uid import UID
from .....io.address import Address
from ....abstract.node_service_interface import NodeServiceInterface


# Inner Payload message using Pydantic.
class Payload(BaseModel):
    class Config:
        orm_mode = True


class RequestPayload(Payload):
    pass


class ReplyPayload(Payload):
    pass


class SyftMessage(ImmediateSyftMessage):
    __attr_allowlist__ = ["id", "address", "reply_to", "reply", "msg_id", "kwargs"]

    signed_type = SignedMessage
    request_payload_type = RequestPayload
    reply_payload_type = ReplyPayload

    def __init__(
        self,
        address: Address,
        kwargs: Union[List, Dict[str, Any]] = {},
        msg_id: Optional[UID] = None,
        reply_to: Optional[Address] = None,
        reply: bool = False,
    ) -> None:
        super().__init__(address=address, msg_id=msg_id)
        self.reply_to = reply_to
        self.reply = reply
        self.kwargs = kwargs

    @property
    def payload(self) -> Payload:
        kwargs_dict = {}

        if hasattr(self.kwargs, "upcast"):
            kwargs_dict = self.kwargs.upcast()  # type: ignore
        else:
            kwargs_dict = self.kwargs  # type: ignore

        # If it's not a reply message then load kwargs as a proper request payload.
        if not self.reply:
            return self.request_payload_type(**kwargs_dict)
        # If it's a reply message, then load kwargs as a proper reply payload.
        else:
            return self.reply_payload_type(**kwargs_dict)

    def run(
        self, node: NodeServiceInterface, verify_key: Optional[VerifyKey] = None
    ) -> Union[None, Payload]:
        raise NotImplementedError
