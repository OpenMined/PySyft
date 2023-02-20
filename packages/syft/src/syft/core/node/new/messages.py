# stdlib
from enum import Enum
from typing import Optional
from typing import Type

# relative
from ....core.node.common.node_table.syft_object import SYFT_OBJECT_VERSION_1
from ....core.node.common.node_table.syft_object import SyftObject
from ...common.serde.serializable import serializable
from ...common.uid import UID
from .credentials import SyftVerifyKey
from .document_store import StorePartition
from .request import DateTime
from .service import AbstractService
from .transforms import transform


class MessageStatus(Enum):
    UNDELIVERED = 0
    DELIVERED = 1


class MessageExpiryStatus(Enum):
    AUTO = 0
    NEVER = 1


@serializable(recursive_serde=True)
class DocumentLink(SyftObject):
    partition_type: Type[StorePartition]
    node_uid: UID
    service: Type[AbstractService]
    document_uid: UID


@serializable(recursive_serde=True)
class Message(SyftObject):
    __canonical_name__ = "Message"
    __version__ = SYFT_OBJECT_VERSION_1

    subject: str
    node_uid: UID
    from_user_verify_key: SyftVerifyKey
    to_user_verify_key: SyftVerifyKey
    created_at: DateTime
    status: MessageStatus = MessageStatus.PENDING
    document_link: Optional[DocumentLink]

    __attr_searchable__ = [
        "from_user_verify_key",
        "to_user_verify_key",
        "status",
    ]
    __attr_unique__ = ["id"]


class MessageDelivered(Message):
    status: MessageStatus = MessageStatus.DELIVERED


@transform(Message, MessageDelivered)
def msg_to_msg_delivered():
    return []
