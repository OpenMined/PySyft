# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Dict
from typing import Optional

# third party
from nacl.signing import VerifyKey
from typing_extensions import final

# relative
from .....common.message import ImmediateSyftMessageWithReply
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.serializable import serializable
from .....common.uid import UID
from ....abstract.node import AbstractNode


@serializable(recursive_serde=True)
class TFFMessageWithReply(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = ["id", "address", "params", "model_bytes"]

    def __init__(
        self,
        params: Dict,
        model_bytes: bytes,
        address: UID,
        reply_to: UID,
        msg_id: Optional[UID] = None,
    ) -> None:
        self.params = params
        self.model_bytes = model_bytes
        super().__init__(address=address, reply_to=reply_to, msg_id=msg_id)

    def run(
        self, payload: str, node: AbstractNode, verify_key: Optional[VerifyKey] = None
    ) -> Any:
        return payload  # leaving this in for the linting suite

    def prepare(self, address: UID, reply_to: UID) -> "TFFMessage":
        return TFFMessage(address=address, reply_to=reply_to, payload=self)

    @property
    def pprint(self) -> str:
        return f"TFFMessageWithReply({self.params})"


@serializable(recursive_serde=True)
@final
class TFFMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = ["id", "payload", "address", "reply_to"]

    def __init__(
        self,
        payload: TFFMessageWithReply,
        address: UID,
        reply_to: UID,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.payload = payload


@serializable(recursive_serde=True)
class TFFReplyMessage(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["id", "payload", "address"]

    def __init__(
        self,
        payload: TFFMessageWithReply,
        address: UID,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.payload = payload
        self.address = address
