# stdlib

# relative
from ...serde.serializable import serializable
from ...store.db.db import DBManager
from ...types.errors import SyftException
from ...types.uid import UID
from ..context import AuthedServiceContext
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import SERVICE_TO_TYPES
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from ..user.user_roles import DATA_OWNER_ROLE_LEVEL
from .image_registry import SyftImageRegistry
from .image_registry_stash import SyftImageRegistryStash

__all__ = ["SyftImageRegistryService"]


@serializable(canonical_name="SyftImageRegistryService", version=1)
class SyftImageRegistryService(AbstractService):
    stash: SyftImageRegistryStash

    def __init__(self, store: DBManager) -> None:
        self.stash = SyftImageRegistryStash(store=store)

    @service_method(
        path="image_registry.add",
        name="add",
        roles=DATA_OWNER_ROLE_LEVEL,
        unwrap_on_success=False,
    )
    def add(
        self,
        context: AuthedServiceContext,
        url: str,
    ) -> SyftSuccess:
        try:
            registry = SyftImageRegistry.from_url(url)
        except Exception as e:
            raise SyftException(public_message=f"Failed to create registry. {e}")

        stored_registry = self.stash.set(context.credentials, registry).unwrap()
        return SyftSuccess(
            message=f"Image Registry ID: {registry.id} created successfully",
            value=stored_registry,
        )

    @service_method(
        path="image_registry.delete",
        name="delete",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def delete(
        self,
        context: AuthedServiceContext,
        uid: UID | None = None,
        url: str | None = None,
    ) -> SyftSuccess:
        # TODO - we need to make sure that there are no workers running an image bound to this registry

        # if url is provided, get uid from url
        if url:
            self.stash.delete_by_url(context.credentials, url).unwrap()
            return SyftSuccess(
                message=f"Image Registry URL: {url} successfully deleted."
            )

        # if uid is provided, delete by uid
        if uid:
            self.stash.delete_by_uid(context.credentials, uid).unwrap()
            return SyftSuccess(
                message=f"Image Registry ID: {uid} successfully deleted."
            )
        else:
            raise SyftException(message="Either UID or URL must be provided.")

    @service_method(
        path="image_registry.get_all",
        name="get_all",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def get_all(
        self,
        context: AuthedServiceContext,
    ) -> list[SyftImageRegistry]:
        return self.stash.get_all(context.credentials).unwrap()

    @service_method(
        path="image_registry.get_by_id",
        name="get_by_id",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def get_by_id(self, context: AuthedServiceContext, uid: UID) -> SyftImageRegistry:
        return self.stash.get_by_uid(context.credentials, uid).unwrap()


TYPE_TO_SERVICE[SyftImageRegistry] = SyftImageRegistryService
SERVICE_TO_TYPES[SyftImageRegistryService].update({SyftImageRegistry})
