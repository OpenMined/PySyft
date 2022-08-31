# stdlib
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Type

# third party
from typing_extensions import final

# relative
from ....common.message import ImmediateSyftMessageWithoutReply
from ....common.serde.serializable import serializable
from ....common.serde.util import deserialize_type
from ....common.serde.util import serialize_type
from ....common.uid import UID
from ....io.address import Address


class UnknownPrivateException(Exception):
    pass


@final
@serializable(recursive_serde=True)
class ExceptionMessage(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = [
        "id",
        "address",
        "msg_id_causing_exception",
        "exception_msg",
        "exception_type",
    ]

    __serde_overrides__: Dict[str, Sequence[Callable]] = {
        "exception_type": (serialize_type, deserialize_type)
    }

    def __init__(
        self,
        address: Address,
        msg_id_causing_exception: UID,
        exception_type: Type,
        exception_msg: str,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.msg_id_causing_exception = msg_id_causing_exception
        self.exception_type = exception_type
        self.exception_msg = exception_msg
