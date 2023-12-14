# stdlib
from typing import Union

# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...types.uid import UID
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
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
        path="worker_image.add_image_registry",
        name="add_image_registry",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def add_image_registry(
        self,
        context: AuthedServiceContext,
        url: str,
    ) -> Union[SyftSuccess, SyftError]:
        registry = SyftImageRegistry.from_url(url)
        res = self.stash.set(context.credentials, registry)

        if res.is_err():
            return SyftError(message=res.err())

        return SyftSuccess(
            message=f"Image registry <id: {registry.id}> successfully added."
        )

    @service_method(
        path="worker_image.delete_image_registry",
        name="delete_image_registry",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def delete_image_registry(
        self,
        context: AuthedServiceContext,
        uid: UID = None,
        url: str = None,
    ) -> Union[SyftSuccess, SyftError]:
        # FIXME - we need to make sure that there are no workers running an image bound to this registry

        # if url is provided, get uid from url
        if url:
            res = self.stash.delete_by_url(context.credentials, url)
            if res.is_err():
                return SyftError(message=res.err())
            return SyftSuccess(
                message=f"Image registry <url: {url}> successfully deleted."
            )

        # if uid is provided, delete by uid
        if uid:
            res = self.stash.delete_by_uid(context.credentials, uid)
            if res.is_err():
                return SyftError(message=res.err())
            return SyftSuccess(
                message=f"Image registry <id: {uid}> successfully deleted."
            )
        else:
            return SyftError(message="Either UID or URL must be provided.")
