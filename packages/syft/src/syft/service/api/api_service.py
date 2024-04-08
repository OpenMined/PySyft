# stdlib
from typing import Any

# third party
from pydantic import ValidationError

# relative
from ...serde.serializable import serializable
from ...service.action.action_endpoint import CustomEndpointActionObject
from ...service.action.action_object import ActionObject
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
from .api import PrivateAPIEndpoint
from .api import PublicAPIEndpoint
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

        endpoint_exists = self.stash.path_exists(context.credentials, new_endpoint.path)

        if endpoint_exists.is_err():
            return SyftError(message=endpoint_exists.err())

        if endpoint_exists.is_ok() and endpoint_exists.ok():
            return SyftError(
                message="An API endpoint already exists at the given path."
            )

        result = self.stash.upsert(context.credentials, endpoint=new_endpoint)
        if result.is_err():
            return SyftError(message=result.err())

        result = result.ok()
        action_obj = ActionObject.from_obj(
            id=result.id,
            syft_action_data=CustomEndpointActionObject(endpoint_id=result.id),
            syft_node_location=context.node.id,
            syft_client_verify_key=context.credentials,
        )
        action_service = context.node.get_service("actionservice")
        res = action_service.set(context=context, action_object=action_obj)
        if res.is_err():
            return SyftError(message=res.err())

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
        mock_function: PublicAPIEndpoint | None = None,
        private_function: PrivateAPIEndpoint | None = None,
        hide_definition: bool | None = None,
    ) -> SyftSuccess | SyftError:
        """Updates an specific API endpoint."""

        endpoint_result = self.stash.get_by_path(context.credentials, endpoint_path)

        if endpoint_result.is_err():
            return SyftError(message=endpoint_result.err())

        if not endpoint_result.ok():
            return SyftError(message=f"Enpoint at path {endpoint_path} doesn't exist")

        endpoint: TwinAPIEndpoint = endpoint_result.ok()

        if not (mock_function or private_function or (hide_definition is not None)):
            return SyftError(
                message='Either "mock_function","private_function" or "hide_definition" are required.'
            )

        updated_mock = (
            mock_function if mock_function is not None else endpoint.mock_function
        )
        updated_private = (
            private_function
            if private_function is not None
            else endpoint.private_function
        )

        try:
            endpoint_update = UpdateTwinAPIEndpoint(
                path=endpoint_path,
                mock_function=updated_mock,
                private_function=updated_private,
            )
        except ValidationError as e:
            return SyftError(message=str(e))

        endpoint.mock_function = endpoint_update.mock_function
        endpoint.private_function = endpoint_update.private_function
        endpoint.signature = updated_mock.signature
        view_access = (
            not hide_definition
            if hide_definition is not None
            else endpoint.mock_function.view_access
        )
        endpoint.mock_function.view_access = view_access
        # Check if the endpoint has a private function
        if endpoint.private_function:
            endpoint.private_function.view_access = view_access

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

        result = self.stash.get_by_path(context.credentials, endpoint_path)

        if result.is_err():
            return SyftError(message=result.err())

        endpoint = result.ok()
        if not endpoint:
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
        result = self.stash.get_by_path(context.node.verify_key, path)
        if result.is_err():
            return SyftError(message=result.err())
        api_endpoint = result.ok()

        return api_endpoint.to(TwinAPIEndpointView, context=context)

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
        admin_key = context.node.get_service("userservice").admin_verify_key()
        result = self.stash.get_all(admin_key)
        if result.is_err():
            return SyftError(message=result.err())

        all_api_endpoints = result.ok()
        api_endpoint_view = []
        for api_endpoint in all_api_endpoints:
            api_endpoint_view.append(
                api_endpoint.to(TwinAPIEndpointView, context=context)
            )

        return api_endpoint_view

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
        if isinstance(custom_endpoint, SyftError):
            return custom_endpoint
        return custom_endpoint.exec(context, *args, **kwargs)

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
        if isinstance(custom_endpoint, SyftError):
            return custom_endpoint
        return custom_endpoint.exec_mock_function(context, *args, **kwargs)

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
            result = custom_endpoint.exec_private_function(context, *args, **kwargs)
        return result

    @service_method(
        path="api.exists",
        name="exists",
    )
    def exists(
        self, context: AuthedServiceContext, uid: UID
    ) -> SyftSuccess | SyftError:
        """Check if an endpoint exists"""
        endpoint = self.get_endpoint_by_uid(context, uid)
        return (
            SyftSuccess(message="Endpoint exists")
            if not isinstance(endpoint, SyftError)
            else endpoint
        )

    # ==== The methods below aren't meant to be called directly by the user, but rather by the node server context. ===
    # Therefore, they are not decorated with @service_method
    def execute_server_side_endpoint_by_id(
        self,
        context: AuthedServiceContext,
        endpoint_uid: UID,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        endpoint = self.get_endpoint_by_uid(context, endpoint_uid)
        if isinstance(endpoint, SyftError):
            return endpoint
        selected_code = endpoint.private_function
        if not selected_code:
            selected_code = endpoint.mock_function

        return endpoint.exec_code(selected_code, context, *args, **kwargs)

    def execute_service_side_endpoint_private_by_id(
        self,
        context: AuthedServiceContext,
        endpoint_uid: UID,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        endpoint = self.get_endpoint_by_uid(context, endpoint_uid)
        if isinstance(endpoint, SyftError):
            return endpoint
        if not endpoint.private_function:
            return SyftError(message="This endpoint does not have a private code")
        return endpoint.exec_code(endpoint.private_function, context, *args, **kwargs)

    def execute_server_side_endpoint_mock_by_id(
        self,
        context: AuthedServiceContext,
        endpoint_uid: UID,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        endpoint = self.get_endpoint_by_uid(context, endpoint_uid)
        if isinstance(endpoint, SyftError):
            return endpoint
        return endpoint.exec_code(endpoint.mock_function, context, *args, **kwargs)

    def get_endpoint_by_uid(
        self, context: AuthedServiceContext, uid: UID
    ) -> TwinAPIEndpoint | SyftError:
        admin_key = context.node.get_service("userservice").admin_verify_key()
        result = self.stash.get_by_uid(admin_key, uid)
        if result.is_err():
            return SyftError(message=result.err())
        return result.ok()

    def get_endpoints(
        self, context: AuthedServiceContext
    ) -> list[TwinAPIEndpoint] | SyftError:
        # TODO: Add ability to specify which roles see which endpoints
        # for now skip auth
        results = self.stash.get_all(context.node.verify_key)
        if results.is_ok():
            return results.ok()
        return SyftError(messages="Unable to get CustomAPIEndpoint")

    def get_code(
        self, context: AuthedServiceContext, endpoint_path: str
    ) -> TwinAPIEndpoint | SyftError:
        result = self.stash.get_by_path(context.node.verify_key, path=endpoint_path)
        if result.is_err():
            return SyftError(
                message=f"CustomAPIEndpoint: {endpoint_path} does not exist"
            )

        if result.is_ok():
            return result.ok()

        return SyftError(message=f"Unable to get {endpoint_path} CustomAPIEndpoint")
