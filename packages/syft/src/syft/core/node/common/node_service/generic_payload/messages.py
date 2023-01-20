# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Dict
from typing import Optional

# third party
from nacl.signing import VerifyKey

# relative
from .....common.message import ImmediateSyftMessageWithReply
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.uid import UID
from .....io.address import Address
from ....abstract.node_service_interface import NodeServiceInterface


class GenericPayloadMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = ["id", "payload", "address", "reply_to", "msg_id"]

    def __init__(
        self,
        payload: GenericPayloadMessageWithReply,
        address: Address,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.payload = payload


class GenericPayloadReplyMessage(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["id", "payload", "address", "msg_id"]

    def __init__(
        self,
        payload: GenericPayloadMessageWithReply,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.payload = payload


class GenericPayloadMessageWithReply:
    __attr_allowlist__ = ["kwargs"]
    message_type = GenericPayloadMessage
    message_reply_type = GenericPayloadReplyMessage

    def __init__(self, kwargs: Optional[Dict[str, Any]] = None) -> None:
        if hasattr(kwargs, "upcast"):
            kwargs = kwargs.upcast()  # type: ignore
        self.kwargs: Dict[str, Any] = kwargs if kwargs is not None else {}

    def run(
        self, node: NodeServiceInterface, verify_key: Optional[VerifyKey] = None
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def to(self, address: Address, reply_to: Address) -> Any:
        return self.message_type(address=address, reply_to=reply_to, payload=self)

    def back_to(self, address: Address) -> Any:
        return self.message_reply_type(address=address, payload=self)
