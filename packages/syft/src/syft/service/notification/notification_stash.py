# relative
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
        if not isinstance(verify_key, SyftVerifyKey | str):
            raise AttributeError("verify_key must be of type SyftVerifyKey or str")
        return self.get_all(
            credentials,
            filters={"to_user_verify_key": verify_key},
        ).unwrap()

    @as_result(StashException)
    def get_all_sent_for_verify_key(
        self, credentials: SyftVerifyKey, verify_key: SyftVerifyKey
    ) -> list[Notification]:
        if not isinstance(verify_key, SyftVerifyKey | str):
            raise AttributeError("verify_key must be of type SyftVerifyKey or str")
        return self.get_all(
            credentials,
            filters={"from_user_verify_key": verify_key},
        ).unwrap()

    @as_result(StashException)
    def get_all_for_verify_key(
        self, credentials: SyftVerifyKey, verify_key: SyftVerifyKey
    ) -> list[Notification]:
        if not isinstance(verify_key, SyftVerifyKey | str):
            raise AttributeError("verify_key must be of type SyftVerifyKey or str")
        return self.get_all(
            credentials,
            filters={"from_user_verify_key": verify_key},
        ).unwrap()

    @as_result(StashException)
    def get_all_by_verify_key_for_status(
        self,
        credentials: SyftVerifyKey,
        verify_key: SyftVerifyKey,
        status: NotificationStatus,
    ) -> list[Notification]:
        if not isinstance(verify_key, SyftVerifyKey | str):
            raise AttributeError("verify_key must be of type SyftVerifyKey or str")
        return self.get_all(
            credentials,
            filters={
                "to_user_verify_key": str(verify_key),
                "status": status.name,
            },
        ).unwrap()

    @as_result(StashException, NotFoundException)
    def get_notification_for_linked_obj(
        self,
        credentials: SyftVerifyKey,
        linked_obj: LinkedObject,
    ) -> Notification:
        return self.get_one(
            credentials,
            filters={
                "linked_obj.id": linked_obj.id,
            },
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
        if not isinstance(verify_key, SyftVerifyKey | str):
            raise AttributeError("verify_key must be of type SyftVerifyKey or str")
        notifications = self.get_all_inbox_for_verify_key(
            credentials,
            verify_key=verify_key,
        ).unwrap()
        for notification in notifications:
            self.delete_by_uid(credentials, uid=notification.id).unwrap()
        return True
