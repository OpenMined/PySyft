# stdlib
from enum import Enum
from typing import List
from typing import Optional

# relative
from ...client.api import APIRegistry
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.linked_obj import LinkedObject
from ...types.datetime import DateTime
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.transforms import TransformContext
from ...types.transforms import add_credentials_for_key
from ...types.transforms import add_node_uid_for_key
from ...types.transforms import generate_id
from ...types.transforms import transform
from ...types.uid import UID


@serializable()
class MessageStatus(Enum):
    UNREAD = 0
    READ = 1


class MessageExpiryStatus(Enum):
    AUTO = 0
    NEVER = 1


@serializable()
class ReplyMessage(SyftObject):
    __canonical_name__ = "ReplyMessage"
    __version__ = SYFT_OBJECT_VERSION_1

    text: str
    target_msg: UID
    id: Optional[UID]
    from_user_verify_key: Optional[SyftVerifyKey]


@serializable()
class Message(SyftObject):
    __canonical_name__ = "Message"
    __version__ = SYFT_OBJECT_VERSION_1

    subject: str
    node_uid: UID
    from_user_verify_key: SyftVerifyKey
    to_user_verify_key: SyftVerifyKey
    created_at: DateTime
    status: MessageStatus = MessageStatus.UNREAD
    linked_obj: Optional[LinkedObject]
    replies: Optional[List[ReplyMessage]] = []

    __attr_searchable__ = [
        "from_user_verify_key",
        "to_user_verify_key",
        "status",
    ]
    __attr_repr_cols__ = ["subject", "status", "created_at", "linked_obj"]

    @property
    def link(self) -> Optional[SyftObject]:
        if self.linked_obj:
            return self.linked_obj.resolve
        return None

    def mark_read(self) -> None:
        api = APIRegistry.api_for(self.node_uid)
        return api.services.messages.mark_as_read(uid=self.id)

    def mark_unread(self) -> None:
        api = APIRegistry.api_for(self.node_uid)
        return api.services.messages.mark_as_unread(uid=self.id)


@serializable()
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
