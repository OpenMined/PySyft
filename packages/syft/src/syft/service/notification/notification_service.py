# stdlib

# relative
from ...serde.serializable import serializable
from ...store.db.db import DBManager
from ...store.document_store_errors import StashException
from ...types.errors import SyftException
from ...types.result import as_result
from ...types.uid import UID
from ..action.action_permissions import ActionObjectREAD
from ..context import AuthedServiceContext
from ..notifier.notifier import NotifierSettings
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import SERVICE_TO_TYPES
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from ..user.user_roles import ADMIN_ROLE_LEVEL
from ..user.user_roles import DATA_SCIENTIST_ROLE_LEVEL
from ..user.user_roles import GUEST_ROLE_LEVEL
from .notification_stash import NotificationStash
from .notifications import CreateNotification
from .notifications import LinkedObject
from .notifications import Notification
from .notifications import NotificationStatus
from .notifications import ReplyNotification


@serializable(canonical_name="NotificationService", version=1)
class NotificationService(AbstractService):
    stash: NotificationStash

    def __init__(self, store: DBManager) -> None:
        self.stash = NotificationStash(store=store)

    @service_method(path="notifications.send", name="send")
    def send(
        self, context: AuthedServiceContext, notification: CreateNotification
    ) -> Notification:
        """Send a new notification"""
        new_notification = notification.to(Notification, context=context)

        # Add read permissions to person receiving this message
        permissions = [
            ActionObjectREAD(
                uid=new_notification.id, credentials=new_notification.to_user_verify_key
            )
        ]

        self.stash.set(
            context.credentials, new_notification, add_permissions=permissions
        ).unwrap()

        context.server.services.notifier.dispatch_notification(
            context, new_notification
        ).unwrap()
        return new_notification

    @service_method(path="notifications.reply", name="reply", roles=GUEST_ROLE_LEVEL)
    def reply(
        self,
        context: AuthedServiceContext,
        reply: ReplyNotification,
    ) -> ReplyNotification:
        msg = self.stash.get_by_uid(
            credentials=context.credentials, uid=reply.target_msg
        ).unwrap(
            public_message=f"The target notification id {reply.target_msg} was not found!"
        )
        reply.from_user_verify_key = context.credentials
        msg.replies.append(reply)
        return self.stash.update(credentials=context.credentials, obj=msg).unwrap(
            public_message="Couldn't add a new notification reply in the target notification"
        )

    @service_method(
        path="notifications.user_settings",
        name="user_settings",
    )
    def user_settings(
        self,
        context: AuthedServiceContext,
    ) -> NotifierSettings:
        return context.server.services.notifier.user_settings(context)

    @service_method(
        path="notifications.settings",
        name="settings",
        roles=ADMIN_ROLE_LEVEL,
    )
    def settings(
        self,
        context: AuthedServiceContext,
    ) -> NotifierSettings:
        return context.server.services.notifier.settings(context).unwrap()

    @service_method(
        path="notifications.activate",
        name="activate",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
        unwrap_on_success=False,
    )
    def activate(
        self,
        context: AuthedServiceContext,
    ) -> Notification:
        return context.server.services.notifier.activate(context).unwrap()

    @service_method(
        path="notifications.deactivate",
        name="deactivate",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
        unwrap_on_success=False,
    )
    def deactivate(
        self,
        context: AuthedServiceContext,
    ) -> SyftSuccess:
        return context.server.services.notifier.deactivate(context).unwrap()

    @service_method(
        path="notifications.get_all",
        name="get_all",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get_all(
        self,
        context: AuthedServiceContext,
    ) -> list[Notification]:
        return self.stash.get_all_inbox_for_verify_key(
            context.credentials,
            verify_key=context.credentials,
        ).unwrap()

    @service_method(
        path="notifications.get_all_sent",
        name="outbox",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get_all_sent(self, context: AuthedServiceContext) -> list[Notification]:
        return self.stash.get_all_sent_for_verify_key(
            context.credentials, context.credentials
        ).unwrap()

    # get_all_read and unread cover the same functionality currently as
    # get_all_for_status. However, there may be more statuses added in the future,
    # so we are keeping the more generic get_all_for_status method.
    @as_result(StashException)
    def get_all_for_status(
        self,
        context: AuthedServiceContext,
        status: NotificationStatus,
    ) -> list[Notification]:
        return self.stash.get_all_by_verify_key_for_status(
            context.credentials, verify_key=context.credentials, status=status
        ).unwrap()

    @service_method(
        path="notifications.get_all_read",
        name="get_all_read",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get_all_read(
        self,
        context: AuthedServiceContext,
    ) -> list[Notification]:
        return self.get_all_for_status(
            context=context,
            status=NotificationStatus.READ,
        ).unwrap()

    @service_method(
        path="notifications.get_all_unread",
        name="get_all_unread",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get_all_unread(
        self,
        context: AuthedServiceContext,
    ) -> list[Notification]:
        return self.get_all_for_status(
            context=context,
            status=NotificationStatus.UNREAD,
        ).unwrap()

    @service_method(path="notifications.mark_as_read", name="mark_as_read")
    def mark_as_read(self, context: AuthedServiceContext, uid: UID) -> Notification:
        return self.stash.update_notification_status(
            context.credentials, uid=uid, status=NotificationStatus.READ
        ).unwrap()

    @service_method(path="notifications.mark_as_unread", name="mark_as_unread")
    def mark_as_unread(self, context: AuthedServiceContext, uid: UID) -> Notification:
        return self.stash.update_notification_status(
            context.credentials, uid=uid, status=NotificationStatus.UNREAD
        ).unwrap()

    @service_method(
        path="notifications.resolve_object",
        name="resolve_object",
        roles=GUEST_ROLE_LEVEL,
    )
    def resolve_object(
        self, context: AuthedServiceContext, linked_obj: LinkedObject
    ) -> Notification:
        service = context.server.get_service(linked_obj.service_type)
        return service.resolve_link(context=context, linked_obj=linked_obj).unwrap()

    @service_method(path="notifications.clear", name="clear", unwrap_on_success=False)
    def clear(self, context: AuthedServiceContext) -> SyftSuccess:
        self.stash.delete_all_for_verify_key(
            credentials=context.credentials, verify_key=context.credentials
        ).unwrap()
        return SyftSuccess(message="Cleared all notifications")

    @as_result(SyftException)
    def filter_by_obj(
        self, context: AuthedServiceContext, obj_uid: UID
    ) -> Notification:
        notifications = self.stash.get_all(context.credentials).unwrap()
        for notification in notifications:
            if (
                notification.linked_obj
                and notification.linked_obj.object_uid == obj_uid
            ):
                return notification
        raise SyftException(public_message="Could not get notifications!!")


TYPE_TO_SERVICE[Notification] = NotificationService
SERVICE_TO_TYPES[NotificationService].update({Notification})
