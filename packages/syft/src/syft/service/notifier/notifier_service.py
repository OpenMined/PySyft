# stdlib

# stdlib
from typing import Optional
from typing import Union

# relative
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
        return SyftError(message="Not Implemented")
        # Set Notifier Model active field to True
        # notifier = stash.get()
        # if not notifier -> create a new one

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
        name="turn_on",
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

    def init_notifier(self) -> Union[SyftSuccess, SyftError]:
        pass
        # Initialize the notifier service
        # This method should be called when the node starts
        # It should check the current state of the notifier and set it up accordingly
        """Get Settings"""

        # TODO: implement this

        # result = self.stash.get_all(context.node.signing_key.verify_key)
        # if result.is_ok():
        #     settings = result.ok()
        #     # check if the settings list is empty
        #     if len(settings) == 0:
        #         return SyftError(message="No settings found")
        #     result = settings[0]
        #     return Ok(result)
        # else:
        #     return SyftError(message=result.err())

        # notifier = self.stash.get()

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
