# stdlib

# stdlib
from typing import Union

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...util.experimental_flags import flags
from ..context import AuthedServiceContext
from ..context import UnauthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import service_method
from .settings import NodeSettings
from .settings import NodeSettingsUpdate
from .settings_stash import SettingsStash


@serializable()
class SettingsService(AbstractService):
    store: DocumentStore
    stash: SettingsStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = SettingsStash(store=store)

    @service_method(path="settings.get", name="get")
    def get(self, context: UnauthedServiceContext) -> Result[Ok, Err]:
        """Get Settings"""
        result = self.stash.get_all(context.node.signing_key.verify_key)
        if result.is_ok():
            settings = result.ok()
            # check if the settings list is empty
            if len(settings) == 0:
                return SyftError(message="No settings found")
            result = settings[0]
            return Ok(result)
        else:
            return SyftError(message=result.err())

    @service_method(path="settings.set", name="set")
    def set(
        self, context: AuthedServiceContext, settings: NodeSettings
    ) -> Result[Ok, Err]:
        """Set a new the Node Settings"""
        print("Here!")
        result = self.stash.set(context.credentials, settings)
        if result.is_ok():
            return result
        else:
            return SyftError(message=result.err())

    @service_method(path="settings.update", name="update")
    def update(
        self, context: AuthedServiceContext, settings: NodeSettingsUpdate
    ) -> Result[Ok, Err]:
        result = self.stash.get_all(context.credentials)
        if result.is_ok():
            current_settings = result.ok()
            if len(current_settings) > 0:
                new_settings = current_settings[0].copy(
                    update=settings.to_dict(exclude_empty=True)
                )
                update_result = self.stash.update(context.credentials, new_settings)
                if update_result.is_ok():
                    return result
                else:
                    return SyftError(message=update_result.err())
            else:
                return SyftError(message="No settings found")
        else:
            return SyftError(message=result.err())

    @service_method(path="settings.allow_guest_signup", name="allow_guest_signup")
    def allow_guest_signup(
        self, context: AuthedServiceContext, enable: bool
    ) -> Union[SyftSuccess, SyftError]:
        """Enable/Disable Registration for Data Scientist or Guest Users."""
        flags.CAN_REGISTER = enable
        method = context.node.get_service_method(SettingsService.update)
        settings = NodeSettingsUpdate(signup_enabled=enable)

        result = method(context=context, settings=settings)

        if result.is_err():
            return SyftError(message=f"Failed to update settings: {result.err()}")

        message = "enabled" if enable else "disabled"
        return SyftSuccess(message=f"Registration feature successfully {message}")
