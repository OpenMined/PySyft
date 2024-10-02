# stdlib
from collections.abc import Callable
from enum import Enum
from typing import cast

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.linked_obj import LinkedObject
from ...types.datetime import DateTime
from ...types.syft_migration import migrate
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.syft_object import SyftObject
from ...types.transforms import TransformContext
from ...types.transforms import add_credentials_for_key
from ...types.transforms import add_server_uid_for_key
from ...types.transforms import generate_id
from ...types.transforms import transform
from ...types.uid import UID
from ..notifier.notifier_enums import NOTIFIERS
from .email_templates import EmailTemplate


@serializable(canonical_name="NotificationStatus", version=1)
class NotificationStatus(Enum):
    UNREAD = 0
    READ = 1


@serializable(canonical_name="NotificationRequestStatus", version=1)
class NotificationRequestStatus(Enum):
    NO_ACTION = 0


class NotificationExpiryStatus(Enum):
    AUTO = 0
    NEVER = 1


@serializable()
class ReplyNotification(SyftObject):
    __canonical_name__ = "ReplyNotification"
    __version__ = SYFT_OBJECT_VERSION_1

    text: str
    target_msg: UID
    id: UID | None = None  # type: ignore[assignment]
    from_user_verify_key: SyftVerifyKey | None = None


@serializable()
class NotificationV1(SyftObject):
    __canonical_name__ = "Notification"
    __version__ = SYFT_OBJECT_VERSION_1

    subject: str
    server_uid: UID
    from_user_verify_key: SyftVerifyKey
    to_user_verify_key: SyftVerifyKey
    created_at: DateTime
    status: NotificationStatus = NotificationStatus.UNREAD
    linked_obj: LinkedObject | None = None
    notifier_types: list[NOTIFIERS] = []
    email_template: type[EmailTemplate] | None = None
    replies: list[ReplyNotification] | None = []

    __attr_searchable__ = [
        "from_user_verify_key",
        "to_user_verify_key",
        "status",
    ]
    __repr_attrs__ = ["subject", "status", "created_at", "linked_obj"]
    __table_sort_attr__ = "Created at"


@serializable()
class Notification(SyftObject):
    __canonical_name__ = "Notification"
    __version__ = SYFT_OBJECT_VERSION_2

    subject: str
    server_uid: UID
    from_user_verify_key: SyftVerifyKey
    to_user_verify_key: SyftVerifyKey
    created_at: DateTime
    status: NotificationStatus = NotificationStatus.UNREAD
    linked_obj: LinkedObject | None = None
    notifier_types: list[NOTIFIERS] = []
    email_template: type[EmailTemplate] | None = None
    replies: list[ReplyNotification] = []

    __attr_searchable__ = [
        "from_user_verify_key",
        "to_user_verify_key",
        "status",
    ]
    __repr_attrs__ = ["subject", "status", "created_at", "linked_obj"]
    __table_sort_attr__ = "Created at"
    __order_by__ = ("created_at", "asc")

    def _repr_html_(self) -> str:
        return f"""
            <div class='syft-request'>
                <h3>Notification</h3>
                <p><strong>ID: </strong>{self.id}</p>
                <p><strong>Subject: </strong>{self.subject}</p>
                <p><strong>Status: </strong>{self.status.name}</p>
                <p><strong>Created at: </strong>{self.created_at}</p>
                <p><strong>Linked object: </strong>{self.linked_obj}</p>
                <p>
            </div>
        """

    @property
    def link(self) -> SyftObject | None:
        if self.linked_obj:
            return self.linked_obj.resolve
        return None

    def _coll_repr_(self) -> dict[str, str]:
        self.linked_obj = cast(LinkedObject, self.linked_obj)
        return {
            "Subject": self.subject,
            "Status": self.determine_status().name.capitalize(),
            "Created at": str(self.created_at),
            "Linked object": f"{self.linked_obj.object_type.__canonical_name__} ({self.linked_obj.object_uid})",
        }

    def mark_read(self) -> None:
        return self.get_api().services.notifications.mark_as_read(uid=self.id)

    def mark_unread(self) -> None:
        return self.get_api().services.notifications.mark_as_unread(uid=self.id)

    def determine_status(self) -> Enum:
        # relative
        from ..request.request import Request

        self.linked_obj = cast(LinkedObject, self.linked_obj)
        if isinstance(self.linked_obj.resolve, Request):
            return self.linked_obj.resolve.status

        return NotificationRequestStatus.NO_ACTION  # type: ignore[unreachable]


@serializable()
class CreateNotification(SyftObject):
    __canonical_name__ = "CreateNotification"
    __version__ = SYFT_OBJECT_VERSION_1

    subject: str
    from_user_verify_key: SyftVerifyKey | None = None  # type: ignore[assignment]
    to_user_verify_key: SyftVerifyKey | None = None  # type: ignore[assignment]
    linked_obj: LinkedObject | None = None
    notifier_types: list[NOTIFIERS] = []
    email_template: type[EmailTemplate] | None = None


def add_msg_creation_time(context: TransformContext) -> TransformContext:
    if context.output is not None:
        context.output["created_at"] = DateTime.now()
    else:
        raise ValueError(f"{context}'s output is None. No transformation happened")
    return context


@transform(CreateNotification, Notification)
def createnotification_to_notification() -> list[Callable]:
    return [
        generate_id,
        add_msg_creation_time,
        add_credentials_for_key("from_user_verify_key"),
        add_server_uid_for_key("server_uid"),
    ]


@migrate(NotificationV1, Notification)
def migrate_nofitication_v1_to_v2() -> list[Callable]:
    return []  # skip migration, no changes in the class
