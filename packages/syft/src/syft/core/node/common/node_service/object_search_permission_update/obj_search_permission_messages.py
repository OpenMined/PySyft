# DOs and Don's of this class:
# - Do NOT use absolute syft imports (i.e. import syft.core...) Use relative ones.
# - Do NOT put multiple imports on the same line (i.e. from <x> import a, b, c). Use separate lines
# - Do sort imports by length
# - Do group imports by where they come from

# stdlib
from typing import Optional

# third party
from nacl.signing import VerifyKey
from typing_extensions import final

# relative
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.serializable import serializable
from .....common.uid import UID
from .....io.address import Address


@serializable(recursive_serde=True)
@final
class ObjectSearchPermissionUpdateMessage(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = [
        "id",
        "address",
        "target_verify_key",
        "target_object_id",
        "add_instead_of_remove",
    ]

    def __init__(
        self,
        add_instead_of_remove: bool,
        target_verify_key: Optional[VerifyKey],
        target_object_id: UID,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)

        self.add_instead_of_remove = add_instead_of_remove
        self.target_verify_key = target_verify_key
        self.target_object_id = target_object_id
