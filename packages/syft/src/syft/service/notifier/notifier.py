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
from typing import TypeVar

# third party
from pydantic import BaseModel
from result import Err
from result import Ok
from result import Result

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...types.syft_migration import migrate
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.syft_object import SyftObject
from ...types.transforms import drop
from ...types.transforms import make_set_default
from ..context import AuthedServiceContext
from ..notification.notifications import Notification
from ..response import SyftError
from ..response import SyftSuccess
from .notifier_enums import NOTIFIERS
from .smtp_client import SMTPClient


class BaseNotifier(BaseModel):
    def send(
        self, target: SyftVerifyKey, notification: Notification
    ) -> SyftSuccess | SyftError:
        return SyftError(message="Not implemented")


TBaseNotifier = TypeVar("TBaseNotifier", bound=BaseNotifier)


@serializable()
class UserNotificationActivity(SyftObject):
    __canonical_name__ = "UserNotificationActivity"
    __version__ = SYFT_OBJECT_VERSION_1
    count: int = 1
    date: datetime = datetime.now()


@serializable(canonical_name="EmailNotifier", version=1)
class EmailNotifier(BaseNotifier):
    smtp_client: SMTPClient
    sender: str = ""

    def __init__(
        self,
        username: str,
        password: str,
        sender: str,
        server: str,
        port: int = 587,
    ) -> None:
        self.sender = sender
        self.smtp_client = SMTPClient(
            server=server,
            port=port,
            username=username,
            password=password,
        )

    @classmethod
    def check_credentials(
        cls,
        username: str,
        password: str,
        server: str,
        port: int = 587,
    ) -> Result[Ok, Err]:
        return SMTPClient.check_credentials(
            server=server,
            port=port,
            username=username,
            password=password,
        )

    def send(
        self, context: AuthedServiceContext, notification: Notification
    ) -> Result[Ok, Err]:
        try:
            user_service = context.server.get_service("userservice")

            receiver = user_service.get_by_verify_key(notification.to_user_verify_key)

            if not receiver.notifications_enabled[NOTIFIERS.EMAIL]:
                return Ok(
                    "Email notifications are disabled for this user."
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

            self.smtp_client.send(
                sender=self.sender, receiver=receiver_email, subject=subject, body=body
            )
            return Ok("Email sent successfully!")
        except Exception:
            return Err(
                "Some notifications failed to be delivered. Please check the health of the mailing server."
            )


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
class NotifierSettings(SyftObject):
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
    ) -> Result[Ok, Err]:
        return self.notifiers[NOTIFIERS.EMAIL].check_credentials(
            server=server,
            port=port,
            username=username,
            password=password,
        )

    def send_notifications(
        self,
        context: AuthedServiceContext,
        notification: Notification,
    ) -> Result[Ok, Err]:
        notifier_objs: list = self.select_notifiers(notification)

        for notifier in notifier_objs:
            result = notifier.send(context, notification)
            if result.err():
                return result

        return Ok("Notification sent successfully!")

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
                        )
                    )
                # If notifier is not email, we just create the notifier object
                # TODO: Add the other notifiers, and its auth methods
                else:
                    notifier_objs.append(self.notifiers[notifier_type]())  # type: ignore[misc]

        return notifier_objs


@migrate(NotifierSettingsV1, NotifierSettings)
def migrate_server_settings_v1_to_current() -> list[Callable]:
    return [
        make_set_default("email_activity", {}),
        make_set_default("email_rate_limit", {}),
    ]


@migrate(NotifierSettings, NotifierSettingsV1)
def migrate_server_settings_v2_to_v1() -> list[Callable]:
    # Use drop function on "notifications_enabled" attrubute
    return [drop(["email_activity"]), drop(["email_rate_limit"])]
