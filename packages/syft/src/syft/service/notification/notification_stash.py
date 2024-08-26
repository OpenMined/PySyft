# stdlib

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ...serde.json_serde import serialize_json
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.db.stash import ObjectStash
from ...store.document_store import PartitionKey
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
@serializable(canonical_name="NotificationSQLStash", version=1)
class NotificationStash(ObjectStash[Notification]):
    def get_all_inbox_for_verify_key(
        self, credentials: SyftVerifyKey, verify_key: SyftVerifyKey
    ) -> Result[list[Notification], str]:
        return self.get_all_by_field(
            credentials, field_name="verify_key", field_value=str(verify_key)
        )

    def get_all_sent_for_verify_key(
        self, credentials: SyftVerifyKey, verify_key: SyftVerifyKey
    ) -> Result[list[Notification], str]:
        return self.get_all_by_field(
            credentials,
            field_name="from_user_verify_key",
            field_value=str(verify_key),
        )

    def get_all_for_verify_key(
        self, credentials: SyftVerifyKey, verify_key: SyftVerifyKey, qks: QueryKeys
    ) -> Result[list[Notification], str]:
        return self.get_all_by_field(
            credentials, field_name="verify_key", field_value=str(verify_key)
        )

    def get_all_by_verify_key_for_status(
        self,
        credentials: SyftVerifyKey,
        verify_key: SyftVerifyKey,
        status: NotificationStatus,
    ) -> Result[list[Notification], str]:
        return self.get_all_by_fields(
            credentials,
            fields={
                "to_user_verify_key": str(verify_key),
                "status": status.value,
            },
        )

    def get_notification_for_linked_obj(
        self,
        credentials: SyftVerifyKey,
        linked_obj: LinkedObject,
    ) -> Result[Notification, str]:
        # TODO does this work?
        return self.get_one_by_fields(
            credentials, fields={"linked_obj": serialize_json(linked_obj)}
        )

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
