# stdlib

# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...types.uid import UID
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import SERVICE_TO_TYPES
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from ..user.user_roles import DATA_OWNER_ROLE_LEVEL
from .image_registry import SyftImageRegistry
from .image_registry_stash import SyftImageRegistryStash

__all__ = ["SyftImageRegistryService"]


@serializable()
class SyftImageRegistryService(AbstractService):
    store: DocumentStore
    stash: SyftImageRegistryStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = SyftImageRegistryStash(store=store)

    @service_method(
        path="image_registry.add",
        name="add",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def add(
        self,
        context: AuthedServiceContext,
        url: str,
    ) -> SyftSuccess | SyftError:
        try:
            registry = SyftImageRegistry.from_url(url)
        except Exception as e:
            return SyftError(message=f"Failed to create registry. {e}")

        res = self.stash.set(context.credentials, registry)

        if res.is_err():
            return SyftError(message=f"Failed to create registry. {res.err()}")

        return SyftSuccess(
            message=f"Image Registry ID: {registry.id} created successfully"
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
    ) -> SyftSuccess | SyftError:
        # TODO - we need to make sure that there are no workers running an image bound to this registry

        # if url is provided, get uid from url
        if url:
            res = self.stash.delete_by_url(context.credentials, url)
            if res.is_err():
                return SyftError(message=res.err())
            return SyftSuccess(
                message=f"Image Registry URL: {url} successfully deleted."
            )

        # if uid is provided, delete by uid
        if uid:
            res = self.stash.delete_by_uid(context.credentials, uid)
            if res.is_err():
                return SyftError(message=res.err())
            return SyftSuccess(
                message=f"Image Registry ID: {uid} successfully deleted."
            )
        else:
            return SyftError(message="Either UID or URL must be provided.")

    @service_method(
        path="image_registry.get_all",
        name="get_all",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def get_all(
        self,
        context: AuthedServiceContext,
    ) -> list[SyftImageRegistry] | SyftError:
        result = self.stash.get_all(context.credentials)
        if result.is_err():
            return SyftError(message=result.err())
        return result

    @service_method(
        path="image_registry.get_by_id",
        name="get_by_id",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def get_by_id(
        self, context: AuthedServiceContext, uid: UID
    ) -> SyftImageRegistry | SyftError:
        result = self.stash.get_by_uid(context.credentials, uid)
        if result.is_err():
            return SyftError(message=result.err())
        return result


TYPE_TO_SERVICE[SyftImageRegistry] = SyftImageRegistryService
SERVICE_TO_TYPES[SyftImageRegistryService].update({SyftImageRegistry})
