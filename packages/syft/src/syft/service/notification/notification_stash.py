# stdlib

# third party

# relative
from ...serde.json_serde import serialize_json
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.db.stash import ObjectStash
from ...store.document_store_errors import NotFoundException
from ...store.document_store_errors import StashException
from ...store.linked_obj import LinkedObject
from ...types.result import as_result
from ...types.uid import UID
from .notifications import Notification
from .notifications import NotificationStatus


@serializable(canonical_name="NotificationSQLStash", version=1)
class NotificationStash(ObjectStash[Notification]):
    @as_result(StashException)
    def get_all_inbox_for_verify_key(
        self, credentials: SyftVerifyKey, verify_key: SyftVerifyKey
    ) -> list[Notification]:
        return self.get_all_by_field(
            credentials, field_name="verify_key", field_value=str(verify_key)
        ).unwrap()

    @as_result(StashException)
    def get_all_sent_for_verify_key(
        self, credentials: SyftVerifyKey, verify_key: SyftVerifyKey
    ) -> list[Notification]:
        return self.get_all_by_field(
            credentials,
            field_name="from_user_verify_key",
            field_value=str(verify_key),
        ).unwrap()

    @as_result(StashException)
    def get_all_for_verify_key(
        self, credentials: SyftVerifyKey, verify_key: SyftVerifyKey
    ) -> list[Notification]:
        return self.get_all_by_field(
            credentials, field_name="verify_key", field_value=str(verify_key)
        ).unwrap()

    @as_result(StashException)
    def get_all_by_verify_key_for_status(
        self,
        credentials: SyftVerifyKey,
        verify_key: SyftVerifyKey,
        status: NotificationStatus,
    ) -> list[Notification]:
        return self.get_all_by_fields(
            credentials,
            fields={
                "to_user_verify_key": str(verify_key),
                "status": status.value,
            },
        ).unwrap()

    @as_result(StashException, NotFoundException)
    def get_notification_for_linked_obj(
        self,
        credentials: SyftVerifyKey,
        linked_obj: LinkedObject,
    ) -> Notification:
        # TODO does this work?
        return self.get_one_by_fields(
            credentials, fields={"linked_obj": serialize_json(linked_obj)}
        ).unwrap()

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
