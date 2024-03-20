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
from .api import CreateTwinAPIEndpoint
from .api import TwinAPIEndpoint
from .api import TwinAPIEndpointView
from .api import UpdateTwinAPIEndpoint
from .api_stash import TwinAPIEndpointStash


@instrument
@serializable()
class APIService(AbstractService):
    store: DocumentStore
    stash: TwinAPIEndpointStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = TwinAPIEndpointStash(store=store)

    @service_method(
        path="api.add",
        name="add",
        roles=ADMIN_ROLE_LEVEL,
    )
    def set(
        self, context: AuthedServiceContext, endpoint: CreateTwinAPIEndpoint
    ) -> SyftSuccess | SyftError:
        """Register an CustomAPIEndpoint."""
        try:
            new_endpoint = endpoint.to(TwinAPIEndpoint)
        except ValueError as e:
            return SyftError(message=str(e))

        existent_endpoint = self.stash.get_by_path(
            context.credentials, new_endpoint.path
        )

        if existent_endpoint.is_err():
            return SyftError(message=existent_endpoint.err())

        if existent_endpoint.is_ok() and existent_endpoint.ok():
            return SyftError(
                message="An API endpoint already exists at the given path."
            )

        result = self.stash.upsert(context.credentials, endpoint=new_endpoint)
        if result.is_err():
            return SyftError(message=result.err())

        return SyftSuccess(message="Endpoint successfully created.")

    @service_method(
        path="api.update",
        name="update",
        roles=ADMIN_ROLE_LEVEL,
    )
    def update(
        self,
        context: AuthedServiceContext,
        endpoint_path: str,
        endpoint_update: UpdateTwinAPIEndpoint,
    ) -> SyftSuccess | SyftError:
        """Updates an specific API endpoint."""

        endpoint_result = self.stash.get_by_path(context.credentials, endpoint_path)

        if endpoint_result.is_err():
            return SyftError(message=endpoint_result.err())

        if not endpoint_result.ok():
            return SyftError(f"Enpoint at path {endpoint_path} doesn't exist")

        endpoint: TwinAPIEndpoint = endpoint_result.ok()

        # TODO: should I use a transform for this?
        # TODO: check signature match and everything else
        # TODO: the current UpdateTwinAPIEndpoint is too simplistic. Should perform all necesary checks

        endpoint.public_code = endpoint_update.public_code
        endpoint.private_code_code = endpoint_update.private_code

        result = self.stash.upsert(context.credentials, endpoint=endpoint)
        if result.is_err():
            return SyftError(message=result.err())

        return SyftSuccess(message="Endpoint successfully updated.")

    @service_method(
        path="api.delete",
        name="delete",
        roles=ADMIN_ROLE_LEVEL,
    )
    def delete(
        self, context: AuthedServiceContext, endpoint_path: str
    ) -> SyftSuccess | SyftError:
        """Deletes an specific API endpoint."""

        endpoint = self.stash.get_by_path(context.credentials, endpoint_path)

        if endpoint.is_err():
            return SyftError(message=endpoint.err())
        if not endpoint.ok():
            return SyftError(message=f"Enpoint at path {endpoint_path} doesn't exist")

        result = self.stash.delete_by_uid(context.credentials, endpoint.id)

        if result.is_err():
            return SyftError(message=result.err())

        return SyftSuccess(message="Endpoint successfully deleted.")

    @service_method(
        path="api.view",
        name="view",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def view(
        self, context: AuthedServiceContext, path: str
    ) -> TwinAPIEndpointView | SyftError:
        """Retrieves an specific API endpoint."""
        context.node = cast(AbstractNode, context.node)
        result = self.stash.get_by_path(context.node.verify_key, path)
        if result.is_err():
            return SyftError(message=result.err())
        api_endpoint = result.ok()

        return api_endpoint.to(TwinAPIEndpointView)

    @service_method(
        path="api.api_endpoints",
        name="api_endpoints",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def api_endpoints(
        self,
        context: AuthedServiceContext,
    ) -> list[TwinAPIEndpointView] | SyftError:
        """Retrieves a list of available API endpoints view available to the user."""
        context.node = cast(AbstractNode, context.node)
        admin_key = context.node.get_service("userservice").admin_verify_key()
        result = self.stash.get_all(admin_key)
        if result.is_err():
            return SyftError(message=result.err())

        all_api_endpoints = result.ok()
        api_endpoint_view = []
        for api_endpoint in all_api_endpoints:
            api_endpoint_view.append(api_endpoint.to(TwinAPIEndpointView))

        return api_endpoint_view

    @service_method(
        path="api.schema",
        name="schema",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def api_schema(self, context: AuthedServiceContext, uid: UID) -> TwinAPIEndpoint:
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
        custom_endpoint = self.get_code(
            context=context,
            endpoint_path=path,
        )
        if not isinstance(custom_endpoint, SyftError):
            context, result = custom_endpoint.exec(context, *args, **kwargs)
        return result

    @service_method(path="api.call_public", name="call_public", roles=GUEST_ROLE_LEVEL)
    def call_public(
        self,
        context: AuthedServiceContext,
        path: str,
        *args: Any,
        **kwargs: Any,
    ) -> SyftSuccess | SyftError:
        """Call a Custom API Method in public mode"""
        custom_endpoint = self.get_code(
            context=context,
            endpoint_path=path,
        )
        if not isinstance(custom_endpoint, SyftError):
            context, result = custom_endpoint.exec_public_code(context, *args, **kwargs)
        return result

    @service_method(
        path="api.call_private", name="call_private", roles=GUEST_ROLE_LEVEL
    )
    def call_private(
        self,
        context: AuthedServiceContext,
        path: str,
        *args: Any,
        **kwargs: Any,
    ) -> SyftSuccess | SyftError:
        """Call a Custom API Method in private mode"""
        custom_endpoint = self.get_code(
            context=context,
            endpoint_path=path,
        )
        if not isinstance(custom_endpoint, SyftError):
            context, result = custom_endpoint.exec_private_code(
                context, *args, **kwargs
            )
        return result

    @service_method(
        path="api.exists",
        name="exists",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def exists(
        self, context: AuthedServiceContext, uid: UID
    ) -> SyftSuccess | SyftError:
        """Check if an endpoint exists"""
        result = self.stash.get_by_uid(context.credentials, uid)
        if result.is_err():
            return SyftError(message=result.err())
        return SyftSuccess(message="Endpoint exists")

    def get_endpoints(
        self, context: AuthedServiceContext
    ) -> list[TwinAPIEndpoint] | SyftError:
        # TODO: Add ability to specify which roles see which endpoints
        # for now skip auth
        context.node = cast(AbstractNode, context.node)
        results = self.stash.get_all(context.node.verify_key)
        if results.is_ok():
            return results.ok()
        return SyftError(messages="Unable to get CustomAPIEndpoint")

    def get_code(
        self, context: AuthedServiceContext, endpoint_path: str
    ) -> TwinAPIEndpoint | SyftError:
        context.node = cast(AbstractNode, context.node)
        result = self.stash.get_by_path(context.node.verify_key, path=endpoint_path)
        if not result.is_ok():
            return SyftError(
                message=f"CustomAPIEndpoint: {endpoint_path} does not exist"
            )
        endpoint = result.ok()
        endpoint = endpoint
        if result:
            return endpoint
        return SyftError(message=f"Unable to get {endpoint_path} CustomAPIEndpoint")
