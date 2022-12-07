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
from .....io.address import Address
from ....abstract.node import AbstractNode


@serializable(recursive_serde=True)
class TFFMessageWithReply:
    __attr_allowlist__ = ["address", "params", "model_bytes"]

    def __init__(self, params: Dict, model_bytes: bytes) -> None:
        # self.stuff = stuff
        # self.id_dataset = id_dataset
        self.params = params
        self.model_bytes = model_bytes

    def run(
        self, payload: str, node: AbstractNode, verify_key: Optional[VerifyKey] = None
    ) -> Any:
        return payload  # leaving this in for the linting suite

    def prepare(self, address: Address, reply_to: Address) -> "TFFMessage":
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
        address: Address,
        reply_to: Address,
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
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.payload = payload
        self.address = address
