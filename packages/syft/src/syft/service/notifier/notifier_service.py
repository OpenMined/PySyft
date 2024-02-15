# stdlib

# stdlib
from typing import Optional
from typing import Union

# relative
from ...abstract_node import AbstractNode
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ..context import AuthedServiceContext
from ..notification.notifications import Notification
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import service_method
from ..user.user_roles import ADMIN_ROLE_LEVEL
from ..user.user_roles import DATA_SCIENTIST_ROLE_LEVEL
from .notifier import NotifierSettings
from .notifier_stash import NotifierStash


@serializable()
class NotifierService(AbstractService):
    store: DocumentStore
    stash: NotifierStash  # Which stash should we use?

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = NotifierStash(store=store)

    @service_method(
        path="notifier.settings", name="notifier_settings", roles=ADMIN_ROLE_LEVEL
    )
    def notifier_settings(  # Maybe just notifier.settings
        self,
        context: AuthedServiceContext,
    ) -> Union[NotifierStash, SyftError]:
        """Get Notifier Settings

        Args:
            context: The request context
        Returns:
            Union[NotifierSettings, SyftError]: Notifier Settings or SyftError
        """
        result = self.stash.get(credentials=context.credentials)
        if result.is_err():
            return SyftError(message="Error getting notifier settings")

        return result.ok()

    @service_method(path="notifier.turn_on", name="turn_on", roles=ADMIN_ROLE_LEVEL)
    def turn_on(
        self, context: AuthedServiceContext, email_token: Optional[str] = None
    ) -> Union[SyftSuccess, SyftError]:
        result = self.stash.get(credentials=context.credentials)

        # 1 -  If something went wrong at db level, return the error
        if result.is_err():
            return SyftError(message=result.err())

        notifier = result.ok()
        # 2 - If email token is not provided and notifier doesn't exist, return an error
        if not email_token and not notifier.email_token:
            return SyftError(message="Email token is required to turn on the notifier")

        # 3 - Activate the notifier
        notifier.active = True

        # 4 - If email token is provided.
        if email_token:
            notifier.email_token = email_token

        result = self.stash.update(credentials=context.credentials, settings=notifier)
        if result.is_err():
            return SyftError(message=result.err())
        return SyftSuccess(message="Notifier turned on")

    @service_method(path="notifier.turn_off", name="turn_off", roles=ADMIN_ROLE_LEVEL)
    def turn_off(
        self,
        context: AuthedServiceContext,
    ) -> Union[SyftSuccess, SyftError]:
        result = self.stash.get(credentials=context.credentials)

        if result.is_err():
            return SyftError(message=result.err())

        notifier = result.ok()
        notifier.active = False
        result = self.stash.update(credentials=context.credentials, settings=notifier)
        if result.is_err():
            return SyftError(message=result.err())
        return SyftSuccess(message="Notifier turned off")

    @service_method(
        path="notifier.enable_notifications",
        name="enable_notifications",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def enable_notifications(
        self,
        context: AuthedServiceContext,
    ) -> Union[SyftSuccess, SyftError]:
        return SyftError(message="Not Implemented")
        # Enable current account notifier notifications
        # (Notifications for this user will still be saved in Notifications Service)
        # Store the current notifications state in the stash

    @service_method(
        path="notifier.disable_notifications",
        name="disable_notifications",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def disable_notifications(
        self,
        context: AuthedServiceContext,
    ) -> Union[SyftSuccess, SyftError]:
        return SyftError(message="Not Implemented")
        # Enable current account  notifier notifications
        # (Notifications for this user will still be saved in Notifications Service)
        # Store the current notifications state in the stash

    @staticmethod
    def init_notifier(
        node: AbstractNode,
        active: bool = False,
        email_token: Optional[str] = None,
    ) -> Union[SyftSuccess, SyftError]:
        """Initialize Notifier for a Node.
        If Notifier already exists, it will return the existing one.
        If not, it will create a new one.

        Args:
            node: Node to initialize the notifier
            active: If notifier should be active
            email_token: Email token to send notifications
        Raises:
            Exception: If something went wrong
        Returns:
            Union: SyftSuccess or SyftError
        """
        try:
            # Create a new NotifierStash since its a static method.
            notifier_stash = NotifierStash(store=node.document_store)
            result = notifier_stash.get(node.signing_key.verify_key)
            if result.is_err():
                raise Exception(f"Could not create notifier: {result}")

            # Get the notifier
            notifier = result.ok()
            # If notifier doesn't exist, create a new one
            if not notifier:
                notifier = NotifierSettings(
                    active=active,
                    email_token=email_token,
                )
                notifier_stash.set(node.signing_key.verify_key, notifier)
        except Exception as e:
            print("Unable to create base notifier", e)

    # This is not a public API.
    # This method is used by other services to dispatch notifications internally
    def dispatch_message(
        self, notification: Notification
    ) -> Union[SyftSuccess, SyftError, None]:
        notifier = self.stash.get()
        # If notifier is active
        if notifier.active:
            resp = notifier.send(notification)
            return resp

        # If notifier isn't active, return None
        return None
