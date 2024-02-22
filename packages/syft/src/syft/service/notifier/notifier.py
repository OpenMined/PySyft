# stdlib

# stdlib
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ...abstract_node import AbstractNode
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ..notification.notifications import Notification
from ..response import SyftError
from ..response import SyftSuccess
from .notifier_enums import NOTIFIERS
from .smtp_client import SMTPClient

DEFAULT_EMAIL_SERVER = "smtp.mailgun.org"


class BaseNotifier:
    EMAIL_SERVER = DEFAULT_EMAIL_SERVER

    def send(
        self, target: SyftVerifyKey, notification: Notification
    ) -> Union[SyftSuccess, SyftError]:
        return SyftError(message="Not implemented")


class EmailNotifier(BaseNotifier):
    smtp_client = SMTPClient
    username: str
    password: str
    server: str
    port: int

    def __init__(
        self,
        username: str,
        password: str,
        server: str = DEFAULT_EMAIL_SERVER,
        port: int = 587,
    ) -> None:
        self.username = username
        self.password = password
        self.server = server
        self.port = port

    @classmethod
    def check_credentials(
        cls,
        username: str,
        password: str,
        server: str = DEFAULT_EMAIL_SERVER,
        port: int = 587,
    ) -> bool:
        return cls.smtp_client.check_credentials(
            server=server,
            port=port,
            username=username,
            password=password,
        )

    @staticmethod
    def check_credentials(
        server: str,
        port: int,
        token: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> bool:
        if token:
            return SMTPClient(
                smtp_server=server, smtp_port=port, access_token=token
            ).check_credentials()
        else:
            return SMTPClient(
                smtp_server=server,
                smtp_port=port,
                username=username,
                password=password,
            ).check_credentials()

    def send(self, node: AbstractNode, notification: Notification) -> Result[Ok, Err]:
        try:
            user_service = node.get_service("userservice")
            sender_email = user_service.get_by_verify_key(
                notification.from_user_verify_key
            ).email
            receiver_email = user_service.get_by_verify_key(
                notification.to_user_verify_key
            ).email

            subject = notification.subject
            body = "Testing email notification!"

            if isinstance(receiver_email, str):
                receiver_email = [receiver_email]

            self.smtp_client.send(
                sender=sender_email, receiver=receiver_email, subject=subject, body=body
            )
            return Ok("Email sent successfully!")
        except Exception as e:
            return Err(f"Error: unable to send email: {e}")


# @serializable()
# @dataclass
# class EmailNotifierSettings:
#     """Email notifier configuration"""
#     server: str
#     smtp_client = SMTPClient
#     username: str
#     password: str
#     server: str = DEFAULT_EMAIL_SERVER
#     port: int = 587
#     notifier = EmailNotifier
#     subscribers = set()


@serializable()
class NotifierSettings(SyftObject):
    __canonical_name__ = "NotifierSettings"
    __version__ = SYFT_OBJECT_VERSION_1
    __repr_attrs__ = [
        "active",
        "email_enabled",
        "sms_enabled",
        "slack_enabled",
        "app_enabled",
    ]
    active: bool = False
    # Flag to identify which notification is enabled
    # For now, consider only the email notification
    # In future, Admin, must be able to have a better
    # control on diff notifications.

    notifiers: Dict[NOTIFIERS, Type[BaseNotifier]] = {
        NOTIFIERS.EMAIL: EmailNotifier,
    }

    notifiers_status: Dict[NOTIFIERS, bool] = {
        NOTIFIERS.EMAIL: True,
        NOTIFIERS.SMS: False,
        NOTIFIERS.SLACK: False,
        NOTIFIERS.APP: False,
    }

    email_server: Optional[str] = DEFAULT_EMAIL_SERVER
    email_port: Optional[int] = 587
    email_username: Optional[str] = ""
    email_password: Optional[str] = ""
    email_subscribers = set()

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
        server: Optional[str] = None,
        port: Optional[int] = None,
    ) -> Result[Ok, Err]:
        return self.notifiers[NOTIFIERS.EMAIL].check_credentials(
            server=server if server else self.email_server,
            port=port if port else self.email_port,
            username=username,
            password=password,
        )

    def send_notifications(
        self,
        node: AbstractNode,
        notification: Notification,
    ) -> Result[Ok, Err]:
        notifier_objs: List = self.select_notifiers(notification)

        for notifier in notifier_objs:
            result = notifier.send(node, notification)
            if result.err():
                return result

        return Ok("Notification sent successfully!")

    def select_notifiers(self, notification: Notification) -> List[BaseNotifier]:
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
                # If notifier is email, we need to pass the token
                if notifier_type == NOTIFIERS.EMAIL:
                    notifier_objs.append(
                        self.notifiers[notifier_type](
                            username=self.email_username, password=self.email_password
                        )
                    )
                # If notifier is not email, we just create the notifier object
                # TODO: Add the other notifiers, and its auth methods
                else:
                    notifier_objs.append(self.notifiers[notifier_type]())

        return notifier_objs
