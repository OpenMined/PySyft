# stdlib
from typing import Any
from typing import cast

# relative
from ...abstract_node import AbstractNode
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...types.uid import UID
from ...util.telemetry import instrument
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import service_method
from ..user.user_roles import ADMIN_ROLE_LEVEL
from ..user.user_roles import DATA_SCIENTIST_ROLE_LEVEL
from ..user.user_roles import GUEST_ROLE_LEVEL
from .api import CustomAPIEndpoint
from .api import CustomAPIView
from .api import UpdateCustomAPIEndpoint
from .api_stash import CustomAPIEndpointStash


@instrument
@serializable()
class APIService(AbstractService):
    store: DocumentStore
    stash: CustomAPIEndpointStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = CustomAPIEndpointStash(store=store)

    @service_method(
        path="api.add",
        name="add",
        roles=ADMIN_ROLE_LEVEL,
    )
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

    @service_method(
        path="api.api_endpoints",
        name="api_endpoints",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def api_endpoints(
        self,
        context: AuthedServiceContext,
    ) -> list[CustomAPIEndpoint] | SyftError:
        """Retrieves a list of available API endpoints view available to the user."""
        return SyftError(message="This is not implemented yet.")

    @service_method(
        path="api.update",
        name="update",
        roles=ADMIN_ROLE_LEVEL,
    )
    def update(
        self,
        context: AuthedServiceContext,
        uid: UID,
        updated_api: UpdateCustomAPIEndpoint,
    ) -> SyftSuccess | SyftError:
        """Updates an specific API endpoint."""
        return SyftError(message="This is not implemented yet.")

    @service_method(
        path="api.delete",
        name="delete",
        roles=ADMIN_ROLE_LEVEL,
    )
    def delete(
        self, context: AuthedServiceContext, path: str
    ) -> SyftSuccess | SyftError:
        """Deletes an specific API endpoint."""
        return SyftError(message="This is not implemented yet.")

    @service_method(
        path="api.schema",
        name="schema",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def api_schema(
        self, context: AuthedServiceContext, uid: UID
    ) -> CustomAPIView | SyftError:
        """Show a view of an API endpoint. This must be smart enough to check if
        the user has access to the endpoint."""
        return SyftError(message="This is not implemented yet.")

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
