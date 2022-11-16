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
from ......grid import GridURL
from .....common.serde.serializable import serializable
from ....abstract.node import AbstractNode
from ..generic_payload.messages import GenericPayloadMessage
from ..generic_payload.messages import GenericPayloadMessageWithReply
from ..generic_payload.messages import GenericPayloadReplyMessage


def grid_url_from_kwargs(kwargs: Dict[str, Any]) -> GridURL:
    try:
        if "host_or_ip" in kwargs:
            # old way to send these messages was with host_or_ip
            return GridURL.from_url(str(kwargs["host_or_ip"]))
        elif "grid_url" in kwargs:
            # new way is with grid_url
            return kwargs["grid_url"]
        else:
            raise Exception("kwargs missing host_or_ip or grid_url")
    except Exception as e:
        print(f"Failed to get grid_url from kwargs: {kwargs}. {e}")
        raise e


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
            grid_url = grid_url_from_kwargs(self.kwargs)
            res = requests.get(str(grid_url.with_path("/status")), timeout=0.25)
            return {
                "grid_url": str(grid_url),
                "result": "ping succeeded",
                "status_code": res.status_code,
            }
        except Exception:
            print("Failed to run ping", self.kwargs)
            return {
                "grid_url": str(grid_url),
                "result": "ping failed",
                "status_code": 404,
            }
