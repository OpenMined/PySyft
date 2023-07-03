# stdlib
from typing import List

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...store.linked_obj import LinkedObject
from ...types.datetime import DateTime
from ...types.uid import UID
from ...util.telemetry import instrument
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


@instrument
@serializable()
class NotificationStash(BaseUIDStoreStash):
    object_type = Notification
    settings: PartitionSettings = PartitionSettings(
        name=Notification.__canonical_name__,
        object_type=Notification,
    )

    def get_all_inbox_for_verify_key(
        self, credentials: SyftVerifyKey, verify_key: SyftVerifyKey
    ) -> Result[List[Notification], str]:
        qks = QueryKeys(
            qks=[
                ToUserVerifyKeyPartitionKey.with_obj(verify_key),
            ]
        )
        return self.get_all_for_verify_key(
            credentials=credentials, verify_key=verify_key, qks=qks
        )

    def get_all_sent_for_verify_key(
        self, credentials: SyftVerifyKey, verify_key: SyftVerifyKey
    ) -> Result[List[Notification], str]:
        qks = QueryKeys(
            qks=[
                FromUserVerifyKeyPartitionKey.with_obj(verify_key),
            ]
        )
        return self.get_all_for_verify_key(credentials, verify_key=verify_key, qks=qks)

    def get_all_for_verify_key(
        self, credentials: SyftVerifyKey, verify_key: SyftVerifyKey, qks: QueryKeys
    ) -> Result[List[Notification], str]:
        if isinstance(verify_key, str):
            verify_key = SyftVerifyKey.from_string(verify_key)
        return self.query_all(
            credentials,
            qks=qks,
            order_by=OrderByCreatedAtTimeStampPartitionKey,
        )

    def get_all_by_verify_key_for_status(
        self,
        credentials: SyftVerifyKey,
        verify_key: SyftVerifyKey,
        status: NotificationStatus,
    ) -> Result[List[Notification], str]:
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
        )

    def get_notification_for_linked_obj(
        self,
        credentials: SyftVerifyKey,
        linked_obj: LinkedObject,
    ) -> Result[Notification, str]:
        qks = QueryKeys(
            qks=[
                LinkedObjectPartitionKey.with_obj(linked_obj),
            ]
        )
        return self.query_one(credentials=credentials, qks=qks)

    def update_notification_status(
        self, credentials: SyftVerifyKey, uid: UID, status: NotificationStatus
    ) -> Result[Notification, str]:
        result = self.get_by_uid(credentials, uid=uid)
        if result.is_err():
            return result.err()

        notification = result.ok()
        if notification is None:
            return Err(f"No notification exists for id: {uid}")
        notification.status = status
        return self.update(credentials, obj=notification)

    def delete_all_for_verify_key(
        self, credentials: SyftVerifyKey, verify_key: SyftVerifyKey
    ) -> Result[bool, str]:
        result = self.get_all_inbox_for_verify_key(
            credentials,
            verify_key=verify_key,
        )
        # If result is an error then return the error
        if result.is_err():
            return result

        # get the list of notifications
        notifications = result.ok()

        for notification in notifications:
            result = self.delete_by_uid(credentials, uid=notification.id)
            if result.is_err():
                return result
        return Ok(True)
