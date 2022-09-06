# DOs and Don's of this class:
# - Do NOT use absolute syft imports (i.e. import syft.core...) Use relative ones.
# - Do NOT put multiple imports on the same line (i.e. from <x> import a, b, c). Use separate lines
# - Do sort imports by length
# - Do group imports by where they come from

# stdlib
from typing import List
from typing import Optional
from typing import Type

# third party
from nacl.signing import VerifyKey
from typing_extensions import final

# relative
from ......logger import critical
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.serializable import serializable
from .....common.uid import UID
from .....io.address import Address
from ....abstract.node import AbstractNode
from ..auth import service_auth
from ..node_service import ImmediateNodeServiceWithoutReply


@serializable(recursive_serde=True)
@final
class ReprMessage(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["id", "address"]

    def __init__(self, address: Address, msg_id: Optional[UID] = None):
        super().__init__(address=address, msg_id=msg_id)


class ReprService(ImmediateNodeServiceWithoutReply):
    @staticmethod
    @service_auth(root_only=True)
    def process(node: AbstractNode, msg: ReprMessage, verify_key: VerifyKey) -> None:
        critical(node.__repr__())

    @staticmethod
    def message_handler_types() -> List[Type[ReprMessage]]:
        return [ReprMessage]
