# stdlib

# stdlib
from abc import ABC
from abc import abstractmethod
from typing import Optional
from typing import Union

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ..notification.notifications import Notification
from ..response import SyftError
from ..response import SyftSuccess
from .smtp_client import SMTPClient


class BaseNotifier(ABC):
    @abstractmethod
    def send(
        self, target: SyftVerifyKey, notification: Notification
    ) -> Union[SyftSuccess, SyftError]:
        pass


class BaseEmailNotifier(BaseNotifier):
    # Generic Email settings properties

    api: str
    smtp_client: SMTPClient

    def send(
        self, target: SyftVerifyKey, notification: Notification
    ) -> Union[SyftSuccess, SyftError]:
        # Send the message using the api service
        # user_service = node.serviceaction("userservice") #TODO: fix this
        subs = None  # user_service.get_by_verify_key(target) # TODO: fix this

        target_email = subs.get(BaseEmailNotifier, None)
        email_notification = notification.to_email()
        if target_email:
            self.smtp_client.send(
                subject="subject",
                from_addr="pysyft@example.com",
                to=target_email,
                body=email_notification(),
            )


class PostMarkEmailNotifier(BaseEmailNotifier):
    server: str
    token: Optional[str]

    def __init__(
        self,
        token: Optional[str],
        server: str = "smtp.postmark.com",
    ) -> None:
        self.token = token
        self.server = server
        self.smtp_client = SMTPClient(
            smtp_server=self.server, smtp_port=587, access_token=self.token
        )


@serializable()
class NotifierSettings(SyftObject):
    __canonical_name__ = "NotifierSettings"
    __version__ = SYFT_OBJECT_VERSION_1
    __repr_attrs__ = ["active", "enable_email_notification"]
    active: bool = False
    # Flag to identify which notification is enable
    # For now, consider only the email notification
    # In future, Admin, must be able to have a better
    # control on diff notifications.
    enable_email_notification: bool = True
    email_notifier: Optional[BaseEmailNotifier] = PostMarkEmailNotifier
    email_token: Optional[str] = ""

    def send(
        self, target: SyftVerifyKey, notification: Notification
    ) -> Union[SyftSuccess, SyftError]:
        # The notification definition itself, should have a better context
        # on what kind of notification it wants to trigger.
        # Sometimes we can have more than one, sometimes only one.

        # notifier_objs: List = self.how_to_notify(notification.notifier_types) #TODO: fix this

        for notifier in notification.notifier_types:
            # Sends a notification Email
            if type(notifier, BaseEmailNotifier):
                self.email_notifier.send(notification)
