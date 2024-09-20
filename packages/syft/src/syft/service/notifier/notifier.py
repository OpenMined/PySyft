# .api.services.notifications.settings() is how the server itself would dispatch notifications.
# .api.services.notifications.user_settings() sets if a specific user wants or not to receive notifications.
# Class NotifierSettings holds both pieces of info.
# Users will get notification x where x in {email, slack, sms, app} if three things are set to True:
# 1) .....settings().active
# 2) .....settings().x_enabled
# 2) .....user_settings().x

# stdlib
from collections.abc import Callable
from datetime import datetime
import logging
from typing import Any
from typing import TypeVar

# third party
from pydantic import BaseModel

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...types.errors import SyftException
from ...types.result import as_result
from ...types.syft_migration import migrate
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.syft_object import SYFT_OBJECT_VERSION_3
from ...types.syft_object import SyftObject
from ...types.transforms import drop
from ...types.transforms import make_set_default
from ..context import AuthedServiceContext
from ..notification.notifications import Notification
from ..response import SyftError
from ..response import SyftSuccess
from .notifier_enums import NOTIFICATION_FREQUENCY
from .notifier_enums import NOTIFIERS
from .smtp_client import SMTPClient

logger = logging.getLogger(__name__)


class BaseNotifier(BaseModel):
    @as_result(SyftException)
    def send(
        self, context: AuthedServiceContext, notification: Notification
    ) -> SyftSuccess:
        raise SyftException(public_message="Not implemented")

    @as_result(SyftException)
    def send_batches(
        self, context: AuthedServiceContext, notification_queue: list[Notification]
    ) -> SyftSuccess:
        raise SyftException(public_message="Not implemented")


TBaseNotifier = TypeVar("TBaseNotifier", bound=BaseNotifier)


@serializable()
class UserNotificationActivity(SyftObject):
    __canonical_name__ = "UserNotificationActivity"
    __version__ = SYFT_OBJECT_VERSION_1
    count: int = 1
    date: datetime = datetime.now()


@serializable(canonical_name="EmailNotifier", version=1)
class EmailNotifier(BaseNotifier):
    smtp_client: SMTPClient | None = None
    sender: str = ""

    def __init__(
        self,
        **data: Any,
    ) -> None:
        super().__init__(**data)
        self.sender = data.get("sender", "")
        self.smtp_client = SMTPClient(
            server=data.get("server", ""),
            port=int(data.get("port", 587)),
            username=data.get("username", ""),
            password=data.get("password", ""),
        )

    @classmethod
    def check_credentials(
        cls,
        username: str,
        password: str,
        server: str,
        port: int = 587,
    ) -> bool:
        try:
            SMTPClient.check_credentials(
                server=server,
                port=port,
                username=username,
                password=password,
            )
            return True
        except Exception:
            logger.exception("Credentials validation failed")
            return False

    @as_result(SyftException)
    def send_batches(
        self, context: AuthedServiceContext, notification_queue: list[Notification]
    ) -> SyftSuccess | SyftError:
        subject = None
        receiver_email = None
        sender = None

        notification_sample = notification_queue[0]
        try:
            sender = self.sender
            receiver = context.server.services.user.get_by_verify_key(
                notification_sample.to_user_verify_key
            ).unwrap()
            if not receiver.notifications_enabled[NOTIFIERS.EMAIL]:
                return SyftSuccess(
                    message="Email notifications are disabled for this user."
                )  # TODO: Should we return an error here?
            receiver_email = receiver.email
            if notification_sample.email_template:
                subject = notification_sample.email_template.batched_email_title(
                    notifications=notification_queue, context=context
                )
                body = notification_sample.email_template.batched_email_body(
                    notifications=notification_queue, context=context
                )
            else:
                subject = notification_sample.subject
                body = notification_sample._repr_html_()

            if isinstance(receiver_email, str):
                receiver_email = [receiver_email]

            self.smtp_client.send(  # type: ignore
                sender=sender, receiver=receiver_email, subject=subject, body=body
            )
            message = f"> Sent email: {subject} to {receiver_email}"
            logging.info(message)
            return SyftSuccess(message="Email sent successfully!")
        except Exception as e:
            message = f"> Error sending email: {subject} to {receiver_email} from: {sender}. {e}"
            logger.error(message)
            return SyftError(message="Failed to send an email.")

    @as_result(SyftException)
    def send(
        self, context: AuthedServiceContext, notification: Notification
    ) -> SyftSuccess | SyftError:
        subject = None
        receiver_email = None
        sender = None
        try:
            sender = self.sender
            receiver = context.server.services.user.get_by_verify_key(
                notification.to_user_verify_key
            ).unwrap()
            if not receiver.notifications_enabled[NOTIFIERS.EMAIL]:
                return SyftSuccess(
                    message="Email notifications are disabled for this user."
                )  # TODO: Should we return an error here?
            receiver_email = receiver.email

            if notification.email_template:
                subject = notification.email_template.email_title(
                    notification, context=context
                )
                body = notification.email_template.email_body(
                    notification, context=context
                )
            else:
                subject = notification.subject
                body = notification._repr_html_()

            if isinstance(receiver_email, str):
                receiver_email = [receiver_email]

            self.smtp_client.send(  # type: ignore
                sender=sender, receiver=receiver_email, subject=subject, body=body
            )
            message = f"> Sent email: {subject} to {receiver_email}"
            print(message)
            logging.info(message)
            return SyftSuccess(message="Email sent successfully!")
        except Exception as e:
            message = f"> Error sending email: {subject} to {receiver_email} from: {sender}. {e}"
            logger.error(message)
            return SyftError(message="Failed to send an email.")
            # raise SyftException.from_exception(
            #     exc,
            #     public_message=(
            #         "Some notifications failed to be delivered."
            #         " Please check the health of the mailing server."
            #     ),
            # )


@serializable()
class NotificationPreferences(SyftObject):
    __canonical_name__ = "NotificationPreferences"
    __version__ = SYFT_OBJECT_VERSION_1
    __repr_attrs__ = [
        "email",
        "sms",
        "slack",
        "app",
    ]

    email: bool = False
    sms: bool = False
    slack: bool = False
    app: bool = False


@serializable()
class NotifierSettingsV1(SyftObject):
    __canonical_name__ = "NotifierSettings"
    __version__ = SYFT_OBJECT_VERSION_1
    __repr_attrs__ = [
        "active",
        "email_enabled",
    ]
    active: bool = False

    notifiers: dict[NOTIFIERS, type[TBaseNotifier]] = {
        NOTIFIERS.EMAIL: EmailNotifier,
    }

    notifiers_status: dict[NOTIFIERS, bool] = {
        NOTIFIERS.EMAIL: True,
        NOTIFIERS.SMS: False,
        NOTIFIERS.SLACK: False,
        NOTIFIERS.APP: False,
    }

    email_sender: str | None = ""
    email_server: str | None = ""
    email_port: int | None = 587
    email_username: str | None = ""
    email_password: str | None = ""


@serializable()
class NotifierSettingsV2(SyftObject):
    __canonical_name__ = "NotifierSettings"
    __version__ = SYFT_OBJECT_VERSION_2
    __repr_attrs__ = [
        "active",
        "email_enabled",
    ]
    active: bool = False
    # Flag to identify which notification is enabled
    # For now, consider only the email notification
    # In future, Admin, must be able to have a better
    # control on diff notifications.

    notifiers: dict[NOTIFIERS, type[TBaseNotifier]] = {
        NOTIFIERS.EMAIL: EmailNotifier,
    }

    notifiers_status: dict[NOTIFIERS, bool] = {
        NOTIFIERS.EMAIL: True,
        NOTIFIERS.SMS: False,
        NOTIFIERS.SLACK: False,
        NOTIFIERS.APP: False,
    }

    email_sender: str | None = ""
    email_server: str | None = ""
    email_port: int | None = 587
    email_username: str | None = ""
    email_password: str | None = ""

    email_activity: dict[str, dict[SyftVerifyKey, UserNotificationActivity]] = {}
    email_rate_limit: dict[str, int] = {}


@serializable()
class EmailFrequency(SyftObject):
    __canonical_name__ = "EmailFrequency"
    __version__ = SYFT_OBJECT_VERSION_1

    frequency: NOTIFICATION_FREQUENCY
    start_time: datetime = datetime.now()


@serializable()
class NotifierSettings(SyftObject):
    __canonical_name__ = "NotifierSettings"
    __version__ = SYFT_OBJECT_VERSION_3
    __repr_attrs__ = [
        "active",
        "email_enabled",
    ]
    active: bool = False
    # Flag to identify which notification is enabled
    # For now, consider only the email notification
    # In future, Admin, must be able to have a better
    # control on diff notifications.

    notifiers: dict[NOTIFIERS, type[TBaseNotifier]] = {
        NOTIFIERS.EMAIL: EmailNotifier,
    }

    notifiers_status: dict[NOTIFIERS, bool] = {
        NOTIFIERS.EMAIL: True,
        NOTIFIERS.SMS: False,
        NOTIFIERS.SLACK: False,
        NOTIFIERS.APP: False,
    }

    email_sender: str | None = ""
    email_server: str | None = ""
    email_port: int | None = 587
    email_username: str | None = ""
    email_password: str | None = ""
    email_frequency: dict[str, EmailFrequency] = {}
    email_queue: dict[str, dict[SyftVerifyKey, list[Notification]]] = {}
    email_activity: dict[str, dict[SyftVerifyKey, UserNotificationActivity]] = {}
    email_rate_limit: dict[str, int] = {}

    @property
    def email_enabled(self) -> bool:
        return self.notifiers_status[NOTIFIERS.EMAIL]

    @property
    def sms_enabled(self) -> bool:
        return self.notifiers_status[NOTIFIERS.SMS]

    @property
    def slack_enabled(self) -> bool:
        return self.notifiers_status[NOTIFIERS.SLACK]

    @property
    def app_enabled(self) -> bool:
        return self.notifiers_status[NOTIFIERS.APP]

    def validate_email_credentials(
        self,
        username: str,
        password: str,
        server: str,
        port: int,
    ) -> bool:
        return self.notifiers[NOTIFIERS.EMAIL].check_credentials(
            server=server,
            port=port,
            username=username,
            password=password,
        )

    @as_result(SyftException)
    def send_batched_notification(
        self,
        context: AuthedServiceContext,
        notification_queue: list[Notification],
    ) -> None:
        if len(notification_queue) == 0:
            return None
        notifier_objs: list[BaseNotifier] = self.select_notifiers(notification_queue[0])
        for notifier in notifier_objs:
            notifier.send_batches(
                context=context, notification_queue=notification_queue
            ).unwrap()
        return None

    @as_result(SyftException)
    def send_notifications(
        self,
        context: AuthedServiceContext,
        notification: Notification,
    ) -> int:
        notifier_objs: list[BaseNotifier] = self.select_notifiers(notification)

        for notifier in notifier_objs:
            notifier.send(context=context, notification=notification).unwrap()

        return len(notifier_objs)

    def select_notifiers(self, notification: Notification) -> list[BaseNotifier]:
        """
        Return a list of the notifiers enabled for the given notification"

        Args:
            notification (Notification): The notification object
        Returns:
            List[BaseNotifier]: A list of enabled notifier objects
        """
        notifier_objs = []
        for notifier_type in notification.notifier_types:
            # Check if the notifier is enabled and if it is, create the notifier object
            if (
                self.notifiers_status[notifier_type]
                and self.notifiers[notifier_type] is not None
            ):
                # If notifier is email, we need to pass the parameters
                if notifier_type == NOTIFIERS.EMAIL:
                    notifier_objs.append(
                        self.notifiers[notifier_type](  # type: ignore[misc]
                            username=self.email_username,
                            password=self.email_password,
                            sender=self.email_sender,
                            server=self.email_server,
                            port=self.email_port,
                        )
                    )
                # If notifier is not email, we just create the notifier object
                # TODO: Add the other notifiers, and its auth methods
                else:
                    notifier_objs.append(self.notifiers[notifier_type]())  # type: ignore[misc]

        return notifier_objs


@migrate(NotifierSettingsV1, NotifierSettingsV2)
def migrate_server_settings_v1_to_v2() -> list[Callable]:
    return [
        make_set_default("email_activity", {}),
        make_set_default("email_rate_limit", {}),
    ]


@migrate(NotifierSettingsV2, NotifierSettingsV1)
def migrate_server_settings_v2_to_v1() -> list[Callable]:
    # Use drop function on "notifications_enabled" attrubute
    return [drop(["email_activity"]), drop(["email_rate_limit"])]


@migrate(NotifierSettingsV2, NotifierSettings)
def migrate_server_settings_v2_to_current() -> list[Callable]:
    return [
        make_set_default("email_frequency", {}),
        make_set_default("email_queue", {}),
    ]


@migrate(NotifierSettings, NotifierSettingsV2)
def migrate_server_settings_current_to_v2() -> list[Callable]:
    # Use drop function on "notifications_enabled" attrubute
    return [drop(["email_frequency"]), drop(["email_queue"])]
