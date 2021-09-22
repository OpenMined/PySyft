# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Dict
from typing import Optional

# third party
from nacl.signing import VerifyKey
import requests
from typing_extensions import final

# relative
from .....common.message import ImmediateSyftMessageWithReply
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.serializable import serializable
from .....common.uid import UID
from .....io.address import Address
from ....abstract.node import AbstractNode


@serializable(recursive_serde=True)
class PingMessageWithReply:
    __attr_allowlist__ = ["kwargs"]

    def __init__(self, kwargs: Dict[str, Any]) -> None:
        self.kwargs = kwargs

    def run(
        self, node: AbstractNode, verify_key: Optional[VerifyKey] = None
    ) -> Dict[str, Any]:
        try:
            host_or_ip = str(self.kwargs["host_or_ip"])
            if not host_or_ip.startswith("http"):
                host_or_ip = f"http://{host_or_ip}"
            res = requests.get(f"{host_or_ip}/status")
            return {"host_or_ip": host_or_ip, "status_code": res.status_code}
        except Exception:
            print("Failed to run ping", self.kwargs)
            return {"host_or_ip": host_or_ip, "error": "Error"}

    def to(self, address: Address, reply_to: Address) -> "PingMessage":
        return PingMessage(address=address, reply_to=reply_to, payload=self)

    def back_to(self, address: Address) -> "PingReplyMessage":
        return PingReplyMessage(address=address, payload=self)


@serializable(recursive_serde=True)
@final
class PingMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = ["id", "payload", "address", "reply_to", "msg_id"]

    def __init__(
        self,
        payload: PingMessageWithReply,
        address: Address,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.payload = payload


@serializable(recursive_serde=True)
class PingReplyMessage(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["id", "payload", "address", "msg_id"]

    def __init__(
        self,
        payload: PingMessageWithReply,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.payload = payload
