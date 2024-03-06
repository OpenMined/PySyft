# stdlib
from enum import Enum
from typing import Callable
from typing import List
from typing import Optional
from typing import Type
from typing import cast

# relative
from ...client.api import APIRegistry
from ...client.api import SyftAPI
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.linked_obj import LinkedObject
from ...types.datetime import DateTime
from ...types.syft_migration import migrate
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.syft_object import SyftObject
from ...types.transforms import TransformContext
from ...types.transforms import add_credentials_for_key
from ...types.transforms import add_node_uid_for_key
from ...types.transforms import drop
from ...types.transforms import generate_id
from ...types.transforms import make_set_default
from ...types.transforms import transform
from ...types.uid import UID
from ...util import options
from ...util.colors import SURFACE
from ..notifier.notifier_enums import NOTIFIERS
from .email_templates import EmailTemplate


@serializable()
class NotificationStatus(Enum):
    UNREAD = 0
    READ = 1


@serializable()
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
    id: Optional[UID] = None  # type: ignore[assignment]
    from_user_verify_key: Optional[SyftVerifyKey] = None


@serializable()
class NotificationV1(SyftObject):
    __canonical_name__ = "Notification"
    __version__ = SYFT_OBJECT_VERSION_1

    subject: str
    node_uid: UID
    from_user_verify_key: SyftVerifyKey
    to_user_verify_key: SyftVerifyKey
    created_at: DateTime
    status: NotificationStatus = NotificationStatus.UNREAD
    linked_obj: Optional[LinkedObject] = None
    replies: Optional[List[ReplyNotification]] = []

    __attr_searchable__ = [
        "from_user_verify_key",
        "to_user_verify_key",
        "status",
    ]
    __repr_attrs__ = ["subject", "status", "created_at", "linked_obj"]

    def _repr_html_(self) -> str:
        return f"""
            <style>
            .syft-request {{color: {SURFACE[options.color_theme]}; line-height: 1;}}
            </style>
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
    def link(self) -> Optional[SyftObject]:
        if self.linked_obj:
            return self.linked_obj.resolve
        return None

    def _coll_repr_(self) -> dict[str, str]:
        linked_obj_name: str = ""
        linked_obj_uid: Optional[UID] = None
        if self.linked_obj is not None:
            linked_obj_name = self.linked_obj.object_type.__canonical_name__
            linked_obj_uid = self.linked_obj.object_uid
        return {
            "Subject": self.subject,
            "Status": self.determine_status().name.capitalize(),
            "Created At": str(self.created_at),
            "Linked object": f"{linked_obj_name} ({linked_obj_uid})",
        }

    def mark_read(self) -> None:
        api = APIRegistry.api_for(
            node_uid=self.node_uid, user_verify_key=self.syft_client_verify_key
        )
        if api is None:
            raise ValueError(
                f"Can't access Syft API. You must login to {self.node_uid}"
            )
        return api.services.notifications.mark_as_read(uid=self.id)

    def mark_unread(self) -> None:
        api = APIRegistry.api_for(
            self.node_uid, user_verify_key=self.syft_client_verify_key
        )
        if api is None:
            raise ValueError(
                f"Can't access Syft API. You must login to {self.node_uid}"
            )
        return api.services.notifications.mark_as_unread(uid=self.id)

    def determine_status(self) -> Enum:
        # relative
        from ..request.request import Request

        if self.linked_obj is not None and isinstance(self.linked_obj.resolve, Request):
            return self.linked_obj.resolve.status

        return NotificationRequestStatus.NO_ACTION


@serializable()
class Notification(SyftObject):
    __canonical_name__ = "Notification"
    __version__ = SYFT_OBJECT_VERSION_2

    subject: str
    node_uid: UID
    from_user_verify_key: SyftVerifyKey
    to_user_verify_key: SyftVerifyKey
    created_at: DateTime
    status: NotificationStatus = NotificationStatus.UNREAD
    linked_obj: Optional[LinkedObject]
    notifier_types: List[NOTIFIERS] = []
    email_template: Optional[Type[EmailTemplate]] = None
    replies: Optional[List[ReplyNotification]] = []

    __attr_searchable__ = [
        "from_user_verify_key",
        "to_user_verify_key",
        "status",
    ]
    __repr_attrs__ = ["subject", "status", "created_at", "linked_obj"]

    def _repr_html_(self) -> str:
        return f"""
            <style>
            .syft-request {{color: {SURFACE[options.color_theme]}; line-height: 1;}}
            </style>
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
    def link(self) -> Optional[SyftObject]:
        if self.linked_obj:
            return self.linked_obj.resolve
        return None

    def _coll_repr_(self) -> dict[str, str]:
        self.linked_obj = cast(LinkedObject, self.linked_obj)
        return {
            "Subject": self.subject,
            "Status": self.determine_status().name.capitalize(),
            "Created At": str(self.created_at),
            "Linked object": f"{self.linked_obj.object_type.__canonical_name__} ({self.linked_obj.object_uid})",
        }

    def mark_read(self) -> None:
        api: SyftAPI = cast(
            SyftAPI,
            APIRegistry.api_for(
                self.node_uid, user_verify_key=self.syft_client_verify_key
            ),
        )
        return api.services.notifications.mark_as_read(uid=self.id)

    def mark_unread(self) -> None:
        api: SyftAPI = cast(
            SyftAPI,
            APIRegistry.api_for(
                self.node_uid, user_verify_key=self.syft_client_verify_key
            ),
        )
        return api.services.notifications.mark_as_unread(uid=self.id)

    def determine_status(self) -> Enum:
        # relative
        from ..request.request import Request

        self.linked_obj = cast(LinkedObject, self.linked_obj)
        if isinstance(self.linked_obj.resolve, Request):
            return self.linked_obj.resolve.status

        return NotificationRequestStatus.NO_ACTION  # type: ignore[unreachable]


@migrate(NotificationV1, Notification)
def upgrade_notification_v1_to_v2() -> List[Callable]:
    return [
        make_set_default("notifier_types", []),
    ]


@migrate(Notification, NotificationV1)
def downgrade_notification_v2_to_v1() -> List[Callable]:
    return [
        drop("notifier_types"),
    ]


@serializable()
class CreateNotificationV1(NotificationV1):
    __canonical_name__ = "CreateNotification"
    __version__ = SYFT_OBJECT_VERSION_1

    id: Optional[UID] = None  # type: ignore[assignment]
    node_uid: Optional[UID] = None  # type: ignore[assignment]
    from_user_verify_key: Optional[SyftVerifyKey] = None  # type: ignore[assignment]
    created_at: Optional[DateTime] = None  # type: ignore[assignment]


@serializable()
class CreateNotification(SyftObject):
    __canonical_name__ = "CreateNotification"
    __version__ = SYFT_OBJECT_VERSION_2

    subject: str
    from_user_verify_key: Optional[SyftVerifyKey] = None  # type: ignore[assignment]
    to_user_verify_key: Optional[SyftVerifyKey] = None  # type: ignore[assignment]
    linked_obj: Optional[LinkedObject] = None
    notifier_types: List[NOTIFIERS] = []
    email_template: Optional[Type[EmailTemplate]] = None


@migrate(CreateNotificationV1, CreateNotification)
def upgrade_create_notification_v1_to_v2() -> List[Callable]:
    return [
        make_set_default("notifier_types", []),
    ]


@migrate(CreateNotification, CreateNotificationV1)
def downgrade_create_notification_v2_to_v1() -> List[Callable]:
    return [
        drop("notifier_types"),
    ]


def add_msg_creation_time(context: TransformContext) -> TransformContext:
    if context.output is not None:
        context.output["created_at"] = DateTime.now()
    else:
        print("f{context}'s output is None. No trasformation happened.")
    return context


@transform(CreateNotification, Notification)
def createnotification_to_notification() -> list[Callable]:
    return [
        generate_id,
        add_msg_creation_time,
        add_credentials_for_key("from_user_verify_key"),
        add_node_uid_for_key("node_uid"),
    ]
