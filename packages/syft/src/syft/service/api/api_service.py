# stdlib
from typing import Any
from typing import cast

# relative
from ...abstract_node import AbstractNode
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...util.telemetry import instrument
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import service_method
from ..user.user_roles import GUEST_ROLE_LEVEL
from .api import CustomAPIEndpoint
from .api_stash import CustomAPIEndpointStash


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
    ) -> SyftSuccess | SyftError:
        """Register an CustomAPIEndpoint."""
        result = self.stash.update(context.credentials, endpoint=endpoint)
        if result.is_ok():
            return SyftSuccess(message=f"CustomAPIEndpoint added: {endpoint}")
        return SyftError(
            message=f"Failed to add CustomAPIEndpoint {endpoint}. {result.err()}"
        )

    def get_endpoints(
        self, context: AuthedServiceContext
    ) -> list[CustomAPIEndpoint] | SyftError:
        # TODO: Add ability to specify which roles see which endpoints
        # for now skip auth
        context.node = cast(AbstractNode, context.node)
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
    ) -> SyftSuccess | SyftError:
        """Call a Custom API Method"""
        context.node = cast(AbstractNode, context.node)
        result = self.stash.get_by_path(context.node.verify_key, path=path)
        if not result.is_ok():
            return SyftError(message=f"CustomAPIEndpoint: {path} does not exist")
        custom_endpoint = result.ok()
        custom_endpoint = custom_endpoint[-1]
        if result:
            context, result = custom_endpoint.exec(context, **kwargs)
        return result
