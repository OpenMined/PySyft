# stdlib
import time
from typing import Any
from typing import cast

# third party
from pydantic import ValidationError

# relative
from ...serde.serializable import serializable
from ...service.action.action_endpoint import CustomEndpointActionObject
from ...service.action.action_object import ActionObject
from ...store.db.db import DBManager
from ...store.document_store_errors import NotFoundException
from ...store.document_store_errors import StashException
from ...types.errors import SyftException
from ...types.result import as_result
from ...types.uid import UID
from ..context import AuthedServiceContext
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from ..user.user_roles import ADMIN_ROLE_LEVEL
from ..user.user_roles import DATA_SCIENTIST_ROLE_LEVEL
from ..user.user_roles import GUEST_ROLE_LEVEL
from .api import CreateTwinAPIEndpoint
from .api import Endpoint
from .api import PrivateAPIEndpoint
from .api import PublicAPIEndpoint
from .api import TwinAPIContextView
from .api import TwinAPIEndpoint
from .api import TwinAPIEndpointView
from .api import UpdateTwinAPIEndpoint
from .api_stash import TwinAPIEndpointStash


@serializable(canonical_name="APIService", version=1)
class APIService(AbstractService):
    stash: TwinAPIEndpointStash

    def __init__(self, store: DBManager) -> None:
        self.stash = TwinAPIEndpointStash(store=store)

    @service_method(
        path="api.add", name="add", roles=ADMIN_ROLE_LEVEL, unwrap_on_success=False
    )
    def set(
        self,
        context: AuthedServiceContext,
        endpoint: CreateTwinAPIEndpoint | TwinAPIEndpoint,
    ) -> SyftSuccess:
        """Register an CustomAPIEndpoint."""
        try:
            new_endpoint = None
            if isinstance(endpoint, CreateTwinAPIEndpoint):  # type: ignore
                new_endpoint = endpoint.to(TwinAPIEndpoint)
            elif isinstance(endpoint, TwinAPIEndpoint):  # type: ignore
                new_endpoint = endpoint

            if new_endpoint is None:
                raise SyftException(public_message="Invalid endpoint type.")
        except ValueError as e:
            raise SyftException(public_message=str(e))

        if isinstance(endpoint, CreateTwinAPIEndpoint):
            endpoint_exists = self.stash.path_exists(
                context.credentials, new_endpoint.path
            ).unwrap()
            if endpoint_exists:
                raise SyftException(
                    public_message="An API endpoint already exists at the given path."
                )

        result = self.stash.upsert(context.credentials, obj=new_endpoint).unwrap()
        action_obj = ActionObject.from_obj(
            id=new_endpoint.action_object_id,
            syft_action_data=CustomEndpointActionObject(endpoint_id=result.id),
            syft_server_location=context.server.id,
            syft_client_verify_key=context.credentials,
        )
        context.server.services.action.set_result_to_store(
            context=context,
            result_action_object=action_obj,
            has_result_read_permission=True,
        ).unwrap()

        return SyftSuccess(message="Endpoint successfully created.")

    @service_method(
        path="api.update",
        name="update",
        roles=ADMIN_ROLE_LEVEL,
        unwrap_on_success=False,
    )
    def update(
        self,
        context: AuthedServiceContext,
        endpoint_path: str,
        mock_function: Endpoint | None = None,
        private_function: Endpoint | None = None,
        hide_mock_definition: bool | None = None,
        endpoint_timeout: int | None = None,
    ) -> SyftSuccess:
        """Updates an specific API endpoint."""

        # if any of these are supplied e.g. truthy then keep going otherwise return
        # an error
        # TODO: change to an Update object with autosplat
        if not (
            mock_function
            or private_function
            or (hide_mock_definition is not None)
            or endpoint_timeout
        ):
            raise SyftException(
                public_message='At least one of "mock_function", "private_function", '
                '"hide_mock_definition" or "endpoint_timeout" is required.'
            )

        endpoint = self.stash.get_by_path(context.credentials, endpoint_path).unwrap()

        endpoint_timeout = (
            endpoint_timeout
            if endpoint_timeout is not None
            else endpoint.endpoint_timeout
        )

        updated_mock = (
            mock_function.to(PublicAPIEndpoint)
            if mock_function is not None
            else endpoint.mock_function
        )
        updated_private = (
            private_function.to(PrivateAPIEndpoint)
            if private_function is not None
            else endpoint.private_function
        )

        try:
            endpoint_update = UpdateTwinAPIEndpoint(
                path=endpoint_path,
                mock_function=updated_mock,
                private_function=updated_private,
                endpoint_timeout=endpoint_timeout,
            )
        except ValidationError as e:
            raise SyftException(public_message=str(e))

        endpoint.mock_function = endpoint_update.mock_function
        endpoint.private_function = endpoint_update.private_function
        endpoint.signature = updated_mock.signature
        endpoint.endpoint_timeout = endpoint_update.endpoint_timeout

        if hide_mock_definition is not None:
            view_access = not hide_mock_definition
            endpoint.mock_function.view_access = view_access

        # save changes
        self.stash.upsert(context.credentials, obj=endpoint).unwrap()
        return SyftSuccess(message="Endpoint successfully updated.")

    @service_method(
        path="api.delete",
        name="delete",
        roles=ADMIN_ROLE_LEVEL,
        unwrap_on_success=False,
    )
    def delete(self, context: AuthedServiceContext, endpoint_path: str) -> SyftSuccess:
        """Deletes an specific API endpoint."""
        endpoint = self.stash.get_by_path(context.credentials, endpoint_path).unwrap()
        self.stash.delete_by_uid(context.credentials, endpoint.id).unwrap()
        return SyftSuccess(message="Endpoint successfully deleted.")

    @service_method(
        path="api.view",
        name="view",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def view(self, context: AuthedServiceContext, path: str) -> TwinAPIEndpointView:
        """Retrieves an specific API endpoint."""
        api_endpoint = self.stash.get_by_path(context.server.verify_key, path).unwrap()
        return api_endpoint.to(TwinAPIEndpointView, context=context)

    @service_method(
        path="api.get",
        name="get",
        roles=ADMIN_ROLE_LEVEL,
    )
    def get(self, context: AuthedServiceContext, api_path: str) -> TwinAPIEndpoint:
        """Retrieves an specific API endpoint."""
        return self.stash.get_by_path(context.server.verify_key, api_path).unwrap()

    @service_method(
        path="api.set_state",
        name="set_state",
        roles=ADMIN_ROLE_LEVEL,
    )
    def set_state(
        self,
        context: AuthedServiceContext,
        api_path: str,
        state: dict,
        private: bool = False,
        mock: bool = False,
        both: bool = False,
    ) -> TwinAPIEndpoint:
        """Sets the state of a specific API endpoint."""
        if both:
            private = True
            mock = True
        api_endpoint = self.stash.get_by_path(
            context.server.verify_key, api_path
        ).unwrap()

        if private and api_endpoint.private_function:
            api_endpoint.private_function.state = state
        if mock and api_endpoint.mock_function:
            api_endpoint.mock_function.state = state

        self.stash.upsert(context.credentials, obj=api_endpoint).unwrap()
        return SyftSuccess(message=f"APIEndpoint {api_path} state updated.")

    @service_method(
        path="api.set_settings",
        name="set_settings",
        roles=ADMIN_ROLE_LEVEL,
    )
    def set_settings(
        self,
        context: AuthedServiceContext,
        api_path: str,
        settings: dict,
        private: bool = False,
        mock: bool = False,
        both: bool = False,
    ) -> TwinAPIEndpoint:
        """Sets the settings of a specific API endpoint."""
        if both:
            private = True
            mock = True
        api_endpoint = self.stash.get_by_path(
            context.server.verify_key, api_path
        ).unwrap()

        if private and api_endpoint.private_function:
            api_endpoint.private_function.settings = settings
        if mock and api_endpoint.mock_function:
            api_endpoint.mock_function.settings = settings

        self.stash.upsert(context.credentials, obj=api_endpoint).unwrap()
        return SyftSuccess(message=f"APIEndpoint {api_path} settings updated.")

    @service_method(
        path="api.api_endpoints",
        name="api_endpoints",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def api_endpoints(
        self,
        context: AuthedServiceContext,
    ) -> list[TwinAPIEndpointView]:
        """Retrieves a list of available API endpoints view available to the user."""
        admin_key = context.server.services.user.root_verify_key
        all_api_endpoints = self.stash.get_all(admin_key).unwrap()

        api_endpoint_view = [
            api_endpoint.to(TwinAPIEndpointView, context=context)
            for api_endpoint in all_api_endpoints
        ]

        return api_endpoint_view

    @service_method(
        path="api.call_in_jobs", name="call_in_jobs", roles=GUEST_ROLE_LEVEL
    )
    def call_in_jobs(
        self,
        context: AuthedServiceContext,
        path: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Call a Custom API Method in a Job"""
        return self._call_in_jobs(context, "call", path, *args, **kwargs).unwrap()

    @service_method(
        path="api.call_private_in_jobs",
        name="call_private_in_jobs",
        roles=GUEST_ROLE_LEVEL,
    )
    def call_private_in_jobs(
        self,
        context: AuthedServiceContext,
        path: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Call a Custom API Method in a Job"""
        return self._call_in_jobs(
            context, "call_private", path, *args, **kwargs
        ).unwrap()

    @service_method(
        path="api.call_public_in_jobs",
        name="call_public_in_jobs",
        roles=GUEST_ROLE_LEVEL,
    )
    def call_public_in_jobs(
        self,
        context: AuthedServiceContext,
        path: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Call a Custom API Method in a Job"""
        return self._call_in_jobs(
            context, "call_public", path, *args, **kwargs
        ).unwrap()

    @as_result(SyftException)
    def _call_in_jobs(
        self,
        context: AuthedServiceContext,
        method: str,
        path: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        custom_endpoint = self.get_code(
            context=context,
            endpoint_path=path,
        ).unwrap()
        log_id = UID()
        job = context.server.add_api_endpoint_execution_to_queue(
            context.credentials,
            method,
            path,
            *args,
            worker_pool_name=custom_endpoint.worker_pool_name,
            log_id=log_id,
            **kwargs,
        )

        # relative
        from ..job.job_stash import JobStatus

        # So result is a Job object
        job_id = job.id
        # Question: For a small moment, when job status is updated, it doesn't return the job during the .get() as if
        # it's not in the stash. Then afterwards if appears again. Is this a bug?

        start = time.time()

        # TODO: what can we do here?????
        while (
            job is None
            or job.status == JobStatus.PROCESSING
            or job.status == JobStatus.CREATED
        ):
            job = context.server.services.job.get(context, job_id)
            time.sleep(0.1)
            if (time.time() - custom_endpoint.endpoint_timeout) > start:
                raise SyftException(
                    public_message=(
                        f"Function timed out in {custom_endpoint.endpoint_timeout} seconds. "
                        + f"Get the Job with id: {job_id} to check results."
                    )
                )

        if job.status == JobStatus.COMPLETED:
            return job.result
        elif job.status == JobStatus.ERRORED:
            raise SyftException(
                public_message=f"Function failed to complete: {job.result.message}"
            )
        else:
            raise SyftException(public_message="Function failed to complete.")

    @service_method(
        path="api.get_public_context", name="get_public_context", roles=ADMIN_ROLE_LEVEL
    )
    def get_public_context(
        self, context: AuthedServiceContext, path: str
    ) -> dict[str, Any]:
        """Get specific public api context."""
        custom_endpoint = self.get_code(
            context=context,
            endpoint_path=path,
        ).unwrap()

        return custom_endpoint.mock_function.build_internal_context(context=context).to(
            TwinAPIContextView
        )

    @service_method(
        path="api.get_private_context",
        name="get_private_context",
        roles=ADMIN_ROLE_LEVEL,
    )
    def get_private_context(
        self, context: AuthedServiceContext, path: str
    ) -> dict[str, Any]:
        """Get specific private api context."""
        custom_endpoint = self.get_code(
            context=context,
            endpoint_path=path,
        ).unwrap()

        custom_endpoint.private_function = cast(
            PrivateAPIEndpoint, custom_endpoint.private_function
        )

        return custom_endpoint.private_function.build_internal_context(
            context=context
        ).to(TwinAPIContextView)

    @service_method(path="api.get_all", name="get_all", roles=ADMIN_ROLE_LEVEL)
    def get_all(
        self,
        context: AuthedServiceContext,
    ) -> list[TwinAPIEndpoint]:
        """Get all API endpoints."""
        return self.stash.get_all(context.credentials).unwrap()

    @service_method(path="api.call", name="call", roles=GUEST_ROLE_LEVEL)
    def call(
        self,
        context: AuthedServiceContext,
        path: str,
        *args: Any,
        log_id: UID | None = None,
        **kwargs: Any,
    ) -> SyftSuccess:
        """Call a Custom API Method"""
        custom_endpoint = self.get_code(
            context=context,
            endpoint_path=path,
        ).unwrap()

        exec_result = custom_endpoint.exec(
            context, *args, log_id=log_id, **kwargs
        ).unwrap()
        action_obj = ActionObject.from_obj(exec_result)
        try:
            return context.server.services.action.set_result_to_store(
                context=context,
                result_action_object=action_obj,
                has_result_read_permission=True,
            ).unwrap()
        except Exception as e:
            # stdlib
            import traceback

            raise SyftException(
                public_message=f"Failed to run. {e}, {traceback.format_exc()}"
            )

    @service_method(path="api.call_public", name="call_public", roles=GUEST_ROLE_LEVEL)
    def call_public(
        self,
        context: AuthedServiceContext,
        path: str,
        *args: Any,
        log_id: UID | None = None,
        **kwargs: Any,
    ) -> ActionObject:
        """Call a Custom API Method in public mode"""
        custom_endpoint = self.get_code(
            context=context,
            endpoint_path=path,
        ).unwrap()
        exec_result = custom_endpoint.exec_mock_function(
            context, *args, log_id=log_id, **kwargs
        ).unwrap()

        action_obj = ActionObject.from_obj(exec_result)
        try:
            return context.server.services.action.set_result_to_store(
                context=context,
                result_action_object=action_obj,
                has_result_read_permission=True,
            ).unwrap()
        except Exception as e:
            # stdlib
            import traceback

            raise SyftException(
                public_message=f"Failed to run. {e}, {traceback.format_exc()}"
            )

    @service_method(
        path="api.call_private", name="call_private", roles=GUEST_ROLE_LEVEL
    )
    def call_private(
        self,
        context: AuthedServiceContext,
        path: str,
        *args: Any,
        log_id: UID | None = None,
        **kwargs: Any,
    ) -> ActionObject:
        """Call a Custom API Method in private mode"""
        custom_endpoint = self.get_code(
            context=context,
            endpoint_path=path,
        ).unwrap()

        exec_result = custom_endpoint.exec_private_function(
            context, *args, log_id=log_id, **kwargs
        ).unwrap()

        action_obj = ActionObject.from_obj(exec_result)
        try:
            return context.server.services.action.set_result_to_store(
                context=context, result_action_object=action_obj
            ).unwrap()
        except Exception as e:
            # stdlib
            import traceback

            raise SyftException(
                public_message=f"Failed to run. {e}, {traceback.format_exc()}"
            )

    @service_method(
        path="api.exists",
        name="exists",
    )
    def exists(self, context: AuthedServiceContext, uid: UID) -> SyftSuccess:
        """Check if an endpoint exists"""
        self.get_endpoint_by_uid(context, uid).unwrap()
        return SyftSuccess(message="Endpoint exists")

    # ==== The methods below aren't meant to be called directly by the user, but
    # rather by the server context. ===
    # Therefore, they are not decorated with @service_method
    @as_result(SyftException)
    def execute_server_side_endpoint_by_id(
        self,
        context: AuthedServiceContext,
        endpoint_uid: UID,
        *args: Any,
        log_id: UID | None = None,
        **kwargs: Any,
    ) -> Any:
        endpoint = self.get_endpoint_by_uid(context, endpoint_uid).unwrap()
        selected_code = endpoint.private_function
        if not selected_code:
            selected_code = endpoint.mock_function

        return endpoint.exec_code(
            selected_code, context, *args, log_id=log_id, **kwargs
        ).unwrap()

    @as_result(StashException, NotFoundException, SyftException)
    def execute_service_side_endpoint_private_by_id(
        self,
        context: AuthedServiceContext,
        endpoint_uid: UID,
        *args: Any,
        log_id: UID | None = None,
        **kwargs: Any,
    ) -> Any:
        endpoint = self.get_endpoint_by_uid(context, endpoint_uid).unwrap()
        return endpoint.exec_code(
            endpoint.private_function, context, *args, log_id=log_id, **kwargs
        ).unwrap()

    @as_result(StashException, NotFoundException, SyftException)
    def execute_server_side_endpoint_mock_by_id(
        self,
        context: AuthedServiceContext,
        endpoint_uid: UID,
        *args: Any,
        log_id: UID | None = None,
        **kwargs: Any,
    ) -> Any:
        endpoint = self.get_endpoint_by_uid(context, endpoint_uid).unwrap()
        return endpoint.exec_code(
            endpoint.mock_function, context, *args, log_id=log_id, **kwargs
        ).unwrap()

    @as_result(StashException, NotFoundException)
    def get_endpoint_by_uid(
        self, context: AuthedServiceContext, uid: UID
    ) -> TwinAPIEndpoint:
        admin_key = context.server.services.user.root_verify_key
        return self.stash.get_by_uid(admin_key, uid).unwrap()

    @as_result(StashException)
    def get_endpoints(self, context: AuthedServiceContext) -> list[TwinAPIEndpoint]:
        # TODO: Add ability to specify which roles see which endpoints
        # for now skip auth
        return self.stash.get_all(context.server.verify_key).unwrap()

    @as_result(StashException, NotFoundException)
    def get_code(
        self, context: AuthedServiceContext, endpoint_path: str
    ) -> TwinAPIEndpoint:
        return self.stash.get_by_path(
            context.server.verify_key, path=endpoint_path
        ).unwrap()


TYPE_TO_SERVICE[TwinAPIEndpoint] = APIService
