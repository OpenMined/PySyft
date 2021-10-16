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
from .....common.serde.serializable import serializable
from ....abstract.node import AbstractNode
from ..generic_payload.messages import GenericPayloadMessage
from ..generic_payload.messages import GenericPayloadMessageWithReply
from ..generic_payload.messages import GenericPayloadReplyMessage


@serializable(recursive_serde=True)
@final
class PingMessage(GenericPayloadMessage):
    ...


@serializable(recursive_serde=True)
@final
class PingReplyMessage(GenericPayloadReplyMessage):
    ...


@serializable(recursive_serde=True)
@final
class PingMessageWithReply(GenericPayloadMessageWithReply):
    message_type = PingMessage
    message_reply_type = PingReplyMessage

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
