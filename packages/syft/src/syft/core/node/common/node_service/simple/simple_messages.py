# stdlib
from typing import Any
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
class NodeRunnableMessageWithReply:

    __attr_allowlist__ = ["stuff"]

    def __init__(self, stuff: str) -> None:
        self.stuff = stuff

    def run(self, node: AbstractNode, verify_key: Optional[VerifyKey] = None) -> Any:
        return (
            "Nothing to see here..." + self.stuff
        )  # leaving this in for the test suite

    def prepare(self, address: Address, reply_to: Address) -> "SimpleMessage":
        return SimpleMessage(address=address, reply_to=reply_to, payload=self)


@serializable(recursive_serde=True)
@final
class SimpleMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = ["id", "payload", "address", "reply_to"]

    def __init__(
        self,
        payload: NodeRunnableMessageWithReply,
        address: Address,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.payload = payload


@serializable(recursive_serde=True)
class SimpleReplyMessage(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["id", "address", "payload"]

    def __init__(
        self,
        payload: NodeRunnableMessageWithReply,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.payload = payload
