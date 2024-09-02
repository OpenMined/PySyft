# stdlib

# third party

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.document_store import NewBaseUIDStoreStash
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...store.document_store_errors import NotFoundException
from ...store.document_store_errors import StashException
from ...store.linked_obj import LinkedObject
from ...types.datetime import DateTime
from ...types.result import as_result
from ...types.uid import UID
from .notifications import Notification
from .notifications import NotificationStatus

FromUserVerifyKeyPartitionKey = PartitionKey(
    key="from_user_verify_key", type_=SyftVerifyKey
)
ToUserVerifyKeyPartitionKey = PartitionKey(
    key="to_user_verify_key", type_=SyftVerifyKey
)
StatusPartitionKey = PartitionKey(key="status", type_=NotificationStatus)

OrderByCreatedAtTimeStampPartitionKey = PartitionKey(key="created_at", type_=DateTime)

LinkedObjectPartitionKey = PartitionKey(key="linked_obj", type_=LinkedObject)


@serializable(canonical_name="NotificationStash", version=1)
class NotificationStash(NewBaseUIDStoreStash):
    object_type = Notification
    settings: PartitionSettings = PartitionSettings(
        name=Notification.__canonical_name__,
        object_type=Notification,
    )

    @as_result(StashException)
    def get_all_inbox_for_verify_key(
        self, credentials: SyftVerifyKey, verify_key: SyftVerifyKey
    ) -> list[Notification]:
        qks = QueryKeys(
            qks=[
                ToUserVerifyKeyPartitionKey.with_obj(verify_key),
            ]
        )
        return self.get_all_for_verify_key(
            credentials=credentials, verify_key=verify_key, qks=qks
        ).unwrap()

    @as_result(StashException)
    def get_all_sent_for_verify_key(
        self, credentials: SyftVerifyKey, verify_key: SyftVerifyKey
    ) -> list[Notification]:
        qks = QueryKeys(
            qks=[
                FromUserVerifyKeyPartitionKey.with_obj(verify_key),
            ]
        )
        return self.get_all_for_verify_key(
            credentials, verify_key=verify_key, qks=qks
        ).unwrap()

    @as_result(StashException)
    def get_all_for_verify_key(
        self, credentials: SyftVerifyKey, verify_key: SyftVerifyKey, qks: QueryKeys
    ) -> list[Notification]:
        if isinstance(verify_key, str):
            verify_key = SyftVerifyKey.from_string(verify_key)
        return self.query_all(
            credentials,
            qks=qks,
            order_by=OrderByCreatedAtTimeStampPartitionKey,
        ).unwrap()

    @as_result(StashException)
    def get_all_by_verify_key_for_status(
        self,
        credentials: SyftVerifyKey,
        verify_key: SyftVerifyKey,
        status: NotificationStatus,
    ) -> list[Notification]:
        qks = QueryKeys(
            qks=[
                ToUserVerifyKeyPartitionKey.with_obj(verify_key),
                StatusPartitionKey.with_obj(status),
            ]
        )
        return self.query_all(
            credentials,
            qks=qks,
            order_by=OrderByCreatedAtTimeStampPartitionKey,
        ).unwrap()

    @as_result(StashException, NotFoundException)
    def get_notification_for_linked_obj(
        self,
        credentials: SyftVerifyKey,
        linked_obj: LinkedObject,
    ) -> Notification:
        qks = QueryKeys(
            qks=[
                LinkedObjectPartitionKey.with_obj(linked_obj),
            ]
        )
        return self.query_one(credentials=credentials, qks=qks).unwrap(
            public_message=f"Notifications for Linked Object {linked_obj} not found"
        )

    @as_result(StashException, NotFoundException)
    def update_notification_status(
        self, credentials: SyftVerifyKey, uid: UID, status: NotificationStatus
    ) -> Notification:
        notification = self.get_by_uid(credentials, uid=uid).unwrap()
        notification.status = status
        return self.update(credentials, obj=notification).unwrap()

    @as_result(StashException, NotFoundException)
    def delete_all_for_verify_key(
        self, credentials: SyftVerifyKey, verify_key: SyftVerifyKey
    ) -> bool:
        notifications = self.get_all_inbox_for_verify_key(
            credentials,
            verify_key=verify_key,
        ).unwrap()
        for notification in notifications:
            self.delete_by_uid(credentials, uid=notification.id).unwrap()
        return True
