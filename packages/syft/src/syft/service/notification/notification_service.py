# stdlib

# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...types.uid import UID
from ...util.telemetry import instrument
from ..action.action_permissions import ActionObjectREAD
from ..context import AuthedServiceContext
from ..notifier.notifier import NotifierSettings
from ..response import SyftError
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


@instrument
@serializable()
class NotificationService(AbstractService):
    store: DocumentStore
    stash: NotificationStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = NotificationStash(store=store)

    @service_method(path="notifications.send", name="send")
    def send(
        self, context: AuthedServiceContext, notification: CreateNotification
    ) -> Notification | SyftError:
        """Send a new notification"""
        new_notification = notification.to(Notification, context=context)

        # Add read permissions to person receiving this message
        permissions = [
            ActionObjectREAD(
                uid=new_notification.id, credentials=new_notification.to_user_verify_key
            )
        ]

        result = self.stash.set(
            context.credentials, new_notification, add_permissions=permissions
        )

        notifier_service = context.node.get_service("notifierservice")

        res = notifier_service.dispatch_notification(context, new_notification)
        if isinstance(res, SyftError):
            return res

        if result.is_err():
            return SyftError(message=str(result.err()))
        return result.ok()

    @service_method(path="notifications.reply", name="reply", roles=GUEST_ROLE_LEVEL)
    def reply(
        self,
        context: AuthedServiceContext,
        reply: ReplyNotification,
    ) -> ReplyNotification | SyftError:
        msg = self.stash.get_by_uid(
            credentials=context.credentials, uid=reply.target_msg
        )
        if msg.is_err():
            return SyftError(
                message=f"The target notification id {reply.target_msg} was not found!. Error: {msg.err()}"
            )
        msg = msg.ok()
        reply.from_user_verify_key = context.credentials
        msg.replies.append(reply)
        result = self.stash.update(credentials=context.credentials, obj=msg)

        if result.is_err():
            return SyftError(
                message=f"Couldn't add a new notification reply in the target notification. Error: {result.err()}"
            )

        return result.ok()

    @service_method(
        path="notifications.user_settings",
        name="user_settings",
    )
    def user_settings(
        self,
        context: AuthedServiceContext,
    ) -> NotifierSettings | SyftError:
        notifier_service = context.node.get_service("notifierservice")
        return notifier_service.user_settings(context)

    @service_method(
        path="notifications.settings",
        name="settings",
        roles=ADMIN_ROLE_LEVEL,
    )
    def settings(
        self,
        context: AuthedServiceContext,
    ) -> NotifierSettings | SyftError:
        notifier_service = context.node.get_service("notifierservice")
        result = notifier_service.settings(context)
        return result

    @service_method(
        path="notifications.activate",
        name="activate",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def activate(
        self,
        context: AuthedServiceContext,
    ) -> Notification | SyftError:
        notifier_service = context.node.get_service("notifierservice")
        result = notifier_service.activate(context)
        return result

    @service_method(
        path="notifications.deactivate",
        name="deactivate",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def deactivate(
        self,
        context: AuthedServiceContext,
    ) -> Notification | SyftError:
        notifier_service = context.node.get_service("notifierservice")
        result = notifier_service.deactivate(context)
        return result

    @service_method(
        path="notifications.get_all",
        name="get_all",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get_all(
        self,
        context: AuthedServiceContext,
    ) -> list[Notification] | SyftError:
        result = self.stash.get_all_inbox_for_verify_key(
            context.credentials,
            verify_key=context.credentials,
        )
        if result.err():
            return SyftError(message=str(result.err()))
        notifications = result.ok()
        return notifications

    @service_method(
        path="notifications.get_all_sent",
        name="outbox",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get_all_sent(
        self, context: AuthedServiceContext
    ) -> list[Notification] | SyftError:
        result = self.stash.get_all_sent_for_verify_key(
            context.credentials, context.credentials
        )
        if result.err():
            return SyftError(message=str(result.err()))
        notifications = result.ok()
        return notifications

    # get_all_read and unread cover the same functionality currently as
    # get_all_for_status. However, there may be more statuses added in the future,
    # so we are keeping the more generic get_all_for_status method.
    def get_all_for_status(
        self,
        context: AuthedServiceContext,
        status: NotificationStatus,
    ) -> list[Notification] | SyftError:
        result = self.stash.get_all_by_verify_key_for_status(
            context.credentials, verify_key=context.credentials, status=status
        )
        if result.err():
            return SyftError(message=str(result.err()))
        notifications = result.ok()
        return notifications

    @service_method(
        path="notifications.get_all_read",
        name="get_all_read",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get_all_read(
        self,
        context: AuthedServiceContext,
    ) -> list[Notification] | SyftError:
        return self.get_all_for_status(
            context=context,
            status=NotificationStatus.READ,
        )

    @service_method(
        path="notifications.get_all_unread",
        name="get_all_unread",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get_all_unread(
        self,
        context: AuthedServiceContext,
    ) -> list[Notification] | SyftError:
        return self.get_all_for_status(
            context=context,
            status=NotificationStatus.UNREAD,
        )

    @service_method(path="notifications.mark_as_read", name="mark_as_read")
    def mark_as_read(
        self, context: AuthedServiceContext, uid: UID
    ) -> Notification | SyftError:
        result = self.stash.update_notification_status(
            context.credentials, uid=uid, status=NotificationStatus.READ
        )
        if result.is_err():
            return SyftError(message=str(result.err()))
        return result.ok()

    @service_method(path="notifications.mark_as_unread", name="mark_as_unread")
    def mark_as_unread(
        self, context: AuthedServiceContext, uid: UID
    ) -> Notification | SyftError:
        result = self.stash.update_notification_status(
            context.credentials, uid=uid, status=NotificationStatus.UNREAD
        )
        if result.is_err():
            return SyftError(message=str(result.err()))
        return result.ok()

    @service_method(
        path="notifications.resolve_object",
        name="resolve_object",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def resolve_object(
        self, context: AuthedServiceContext, linked_obj: LinkedObject
    ) -> Notification | SyftError:
        service = context.node.get_service(linked_obj.service_type)
        result = service.resolve_link(context=context, linked_obj=linked_obj)
        if result.is_err():
            return SyftError(message=str(result.err()))
        return result.ok()

    @service_method(path="notifications.clear", name="clear")
    def clear(self, context: AuthedServiceContext) -> SyftError | SyftSuccess:
        result = self.stash.delete_all_for_verify_key(
            credentials=context.credentials, verify_key=context.credentials
        )
        if result.is_ok():
            return SyftSuccess(message="All notifications cleared !!")
        return SyftError(message=str(result.err()))

    def filter_by_obj(
        self, context: AuthedServiceContext, obj_uid: UID
    ) -> Notification | SyftError:
        notifications = self.stash.get_all(context.credentials)
        if notifications.is_err():
            return SyftError(message="Could not get notifications!!")
        for notification in notifications.ok():
            if (
                notification.linked_obj
                and notification.linked_obj.object_uid == obj_uid
            ):
                return notification
        return SyftError(message="Could not get notifications!!")


TYPE_TO_SERVICE[Notification] = NotificationService
SERVICE_TO_TYPES[NotificationService].update({Notification})
