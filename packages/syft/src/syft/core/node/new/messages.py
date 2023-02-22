# stdlib
from enum import Enum
from typing import Optional
from typing import Type

# third party
from typing_extensions import Self

# relative
from ....core.node.common.node_table.syft_object import SYFT_OBJECT_VERSION_1
from ....core.node.common.node_table.syft_object import SyftObject
from ...common.serde.serializable import serializable
from ...common.uid import UID
from .api import APIRegistry
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
class LinkedDocument(SyftObject):
    __canonical_name__ = "LinkedObject"
    __version__ = SYFT_OBJECT_VERSION_1

    node_uid: UID
    document_cname: str
    service_name: str
    object_uid: UID

    def __str__(self) -> str:
        return f"<{self.document_cname}: {self.object_uid}>"

    @property
    def resolve(self) -> SyftObject:
        api = APIRegistry.api_for(node_uid=self.node_uid)
        return api.services.messages.resolve_document(self)

    @staticmethod
    def create_document(
        node_uid: UID,
        object_type: Type[SyftObject],
        object_service: Type[AbstractService],
        object_uid: UID,
    ) -> Self:
        document_cname = object_type.__canonical_name__
        service_name = object_service.__qualname__
        return LinkedDocument(
            node_uid=node_uid,
            document_cname=document_cname,
            service_name=service_name,
            object_uid=object_uid,
        )


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
    linked_document: Optional[LinkedDocument]

    __attr_searchable__ = [
        "from_user_verify_key",
        "to_user_verify_key",
        "status",
    ]
    __attr_unique__ = ["id"]
    __attr_repr_cols__ = ["subject", "status", "created_at", "linked_document"]


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
