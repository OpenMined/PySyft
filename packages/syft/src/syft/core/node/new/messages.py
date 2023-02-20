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
from .request import DateTime
from .service import AbstractService
from .transforms import TransformContext
from .transforms import add_credentials_for_key
from .transforms import add_node_uid_for_key
from .transforms import generate_id
from .transforms import transform


@serializable(recursive_serde=True)
class MessageStatus(Enum):
    UNDELIVERED = 0
    DELIVERED = 1


class MessageExpiryStatus(Enum):
    AUTO = 0
    NEVER = 1


@serializable(recursive_serde=True)
class DocumentLink(SyftObject):
    __canonical_name__ = "DocumentLink"
    __version__ = SYFT_OBJECT_VERSION_1

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
    status: MessageStatus = MessageStatus.UNDELIVERED
    document_link: Optional[DocumentLink]

    __attr_searchable__ = [
        "from_user_verify_key",
        "to_user_verify_key",
        "status",
    ]
    __attr_unique__ = ["id"]
    __attr_repr_cols__ = ["subject", "status"]


@serializable(recursive_serde=True)
class CreateMessage(Message):
    __canonical_name__ = "CreateMessage"
    __version__ = SYFT_OBJECT_VERSION_1

    id: Optional[UID]
    node_uid: Optional[UID]
    from_user_verify_key: Optional[SyftVerifyKey]
    created_at: Optional[DateTime]


def add_msg_creation_time(context: TransformContext) -> TransformContext:
    context.output["created_at"] = DateTime.now()
    return context


@transform(CreateMessage, Message)
def createmessage_to_message():
    return [
        generate_id,
        add_msg_creation_time,
        add_credentials_for_key("from_user_verify_key"),
        add_node_uid_for_key("node_uid"),
    ]
