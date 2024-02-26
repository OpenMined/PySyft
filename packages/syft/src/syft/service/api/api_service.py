# stdlib
from typing import Any
from typing import List
from typing import Union

# third party
from result import Ok
from result import Result

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionSettings
from ...util.telemetry import instrument
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import service_method
from ..user.user_roles import GUEST_ROLE_LEVEL
from .api import CustomAPIEndpoint


@serializable()
class CustomAPIEndpointStash(BaseUIDStoreStash):
    object_type = CustomAPIEndpoint
    settings: PartitionSettings = PartitionSettings(
        name=CustomAPIEndpoint.__canonical_name__, object_type=CustomAPIEndpoint
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def get_by_path(
        self, credentials: SyftVerifyKey, path: str
    ) -> Result[List[CustomAPIEndpoint], str]:
        results = self.get_all(credentials=credentials)
        items = []
        if results.is_ok() and results.ok():
            results = results.ok()
            for result in results:
                if result.path == path:
                    items.append(result)
            return Ok(items)
        else:
            return results

    def update(
        self, credentials: SyftVerifyKey, endpoint: CustomAPIEndpoint
    ) -> Result[CustomAPIEndpoint, str]:
        res = self.check_type(endpoint, CustomAPIEndpoint)
        if res.is_err():
            return res
        result = super().set(
            credentials=credentials, obj=res.ok(), ignore_duplicates=True
        )
        return result


@instrument
@serializable()
class APIService(AbstractService):
    store: DocumentStore
    stash: CustomAPIEndpointStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = CustomAPIEndpointStash(store=store)

    @service_method(path="api.set", name="set")
    def set(
        self, context: AuthedServiceContext, endpoint: CustomAPIEndpoint
    ) -> Union[SyftSuccess, SyftError]:
        """Register an CustomAPIEndpoint."""
        result = self.stash.update(context.credentials, endpoint=endpoint)
        if result.is_ok():
            return SyftSuccess(message=f"CustomAPIEndpoint added: {endpoint}")
        return SyftError(
            message=f"Failed to add CustomAPIEndpoint {endpoint}. {result.err()}"
        )

    def get_endpoints(
        self, context: AuthedServiceContext
    ) -> Union[List[CustomAPIEndpoint], SyftError]:
        # TODO: Add ability to specify which roles see which endpoints
        # for now skip auth
        results = self.stash.get_all(context.node.verify_key)
        if results.is_ok():
            return results.ok()
        return SyftError(messages="Unable to get CustomAPIEndpoint")

    @service_method(path="api.call", name="call", roles=GUEST_ROLE_LEVEL)
    def call(
        self,
        context: AuthedServiceContext,
        path: str,
        *args: Any,
        **kwargs: Any,
    ) -> Union[SyftSuccess, SyftError]:
        """Call a Custom API Method"""
        result = self.stash.get_by_path(context.node.verify_key, path=path)
        if not result.is_ok():
            return SyftError(message=f"CustomAPIEndpoint: {path} does not exist")
        custom_endpoint = result.ok()
        custom_endpoint = custom_endpoint[-1]
        if result:
            context, result = custom_endpoint.exec(context, **kwargs)
        return result
