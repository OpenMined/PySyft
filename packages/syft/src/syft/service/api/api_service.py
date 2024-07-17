# stdlib
import time
from typing import Any
from typing import cast

# third party
from pydantic import ValidationError
from result import Err
from result import Ok

# relative
from ...serde.serializable import serializable
from ...service.action.action_endpoint import CustomEndpointActionObject
from ...service.action.action_object import ActionObject
from ...store.document_store import DocumentStore
from ...types.uid import UID
from ...util.telemetry import instrument
from ..action.action_service import ActionService
from ..context import AuthedServiceContext
from ..response import SyftError
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


@instrument
@serializable(canonical_name="APIService", version=1)
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
        self,
        context: AuthedServiceContext,
        endpoint: CreateTwinAPIEndpoint | TwinAPIEndpoint,
    ) -> SyftSuccess | SyftError:
        """Register an CustomAPIEndpoint."""
        try:
            new_endpoint = None
            if isinstance(endpoint, CreateTwinAPIEndpoint):  # type: ignore
                new_endpoint = endpoint.to(TwinAPIEndpoint)
            elif isinstance(endpoint, TwinAPIEndpoint):  # type: ignore
                new_endpoint = endpoint

            if new_endpoint is None:
                return SyftError(message="Invalid endpoint type.")  # type: ignore
        except ValueError as e:
            return SyftError(message=str(e))

        if isinstance(endpoint, CreateTwinAPIEndpoint):
            endpoint_exists = self.stash.path_exists(
                context.credentials, new_endpoint.path
            )

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
            id=new_endpoint.action_object_id,
            syft_action_data=CustomEndpointActionObject(endpoint_id=result.id),
            syft_server_location=context.server.id,
            syft_client_verify_key=context.credentials,
        )
        action_service = context.server.get_service("actionservice")
        res = action_service.set_result_to_store(
            context=context,
            result_action_object=action_obj,
            has_result_read_permission=True,
        )
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
        mock_function: Endpoint | None = None,
        private_function: Endpoint | None = None,
        hide_mock_definition: bool | None = None,
        endpoint_timeout: int | None = None,
    ) -> SyftSuccess | SyftError:
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
            return SyftError(
                message='At least one of "mock_function", "private_function", '
                '"hide_mock_definition" or "endpoint_timeout" is required.'
            )

        endpoint_result = self.stash.get_by_path(context.credentials, endpoint_path)

        if endpoint_result.is_err():
            return SyftError(message=endpoint_result.err())

        if not endpoint_result.ok():
            return SyftError(message=f"Enpoint at path {endpoint_path} doesn't exist")

        endpoint: TwinAPIEndpoint = endpoint_result.ok()

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
            return SyftError(message=str(e))

        endpoint.mock_function = endpoint_update.mock_function
        endpoint.private_function = endpoint_update.private_function
        endpoint.signature = updated_mock.signature
        endpoint.endpoint_timeout = endpoint_update.endpoint_timeout

        if hide_mock_definition is not None:
            view_access = not hide_mock_definition
            endpoint.mock_function.view_access = view_access

        # save changes
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
        result = self.stash.get_by_path(context.server.verify_key, path)
        if result.is_err():
            return SyftError(message=result.err())
        api_endpoint = result.ok()

        return api_endpoint.to(TwinAPIEndpointView, context=context)

    @service_method(
        path="api.get",
        name="get",
        roles=ADMIN_ROLE_LEVEL,
    )
    def get(
        self, context: AuthedServiceContext, api_path: str
    ) -> TwinAPIEndpoint | SyftError:
        """Retrieves an specific API endpoint."""
        result = self.stash.get_by_path(context.server.verify_key, api_path)
        if result.is_err():
            return SyftError(message=result.err())
        api_endpoint = result.ok()

        return api_endpoint

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
    ) -> TwinAPIEndpoint | SyftError:
        """Sets the state of a specific API endpoint."""
        if both:
            private = True
            mock = True
        result = self.stash.get_by_path(context.server.verify_key, api_path)
        if result.is_err():
            return SyftError(message=result.err())
        api_endpoint = result.ok()

        if private and api_endpoint.private_function:
            api_endpoint.private_function.state = state
        if mock and api_endpoint.mock_function:
            api_endpoint.mock_function.state = state

        result = self.stash.upsert(context.credentials, endpoint=api_endpoint)
        if result.is_err():
            return SyftError(message=result.err())
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
    ) -> TwinAPIEndpoint | SyftError:
        """Sets the settings of a specific API endpoint."""
        if both:
            private = True
            mock = True
        result = self.stash.get_by_path(context.server.verify_key, api_path)
        if result.is_err():
            return SyftError(message=result.err())
        api_endpoint = result.ok()

        if private and api_endpoint.private_function:
            api_endpoint.private_function.settings = settings
        if mock and api_endpoint.mock_function:
            api_endpoint.mock_function.settings = settings

        result = self.stash.upsert(context.credentials, endpoint=api_endpoint)
        if result.is_err():
            return SyftError(message=result.err())
        return SyftSuccess(message=f"APIEndpoint {api_path} settings updated.")

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
        admin_key = context.server.get_service("userservice").admin_verify_key()
        result = self.stash.get_all(admin_key)
        if result.is_err():
            return SyftError(message=result.err())

        all_api_endpoints = result.ok()
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
    ) -> Any | SyftError:
        """Call a Custom API Method in a Job"""
        return self._call_in_jobs(context, "call", path, *args, **kwargs)

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
    ) -> Any | SyftError:
        """Call a Custom API Method in a Job"""
        return self._call_in_jobs(context, "call_private", path, *args, **kwargs)

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
    ) -> Any | SyftError:
        """Call a Custom API Method in a Job"""
        return self._call_in_jobs(context, "call_public", path, *args, **kwargs)

    def _call_in_jobs(
        self,
        context: AuthedServiceContext,
        method: str,
        path: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any | SyftError:
        custom_endpoint = self.get_code(
            context=context,
            endpoint_path=path,
        )
        if isinstance(custom_endpoint, SyftError):
            return custom_endpoint

        result = context.server.add_api_endpoint_execution_to_queue(
            context.credentials,
            method,
            path,
            *args,
            worker_pool=custom_endpoint.worker_pool,
            **kwargs,
        )
        if isinstance(result, SyftError):
            return result
        # relative
        from ..job.job_stash import JobStatus

        # So result is a Job object
        job = result
        job_service = context.server.get_service("jobservice")
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
            job = job_service.get(context, job_id)
            time.sleep(0.1)
            if (time.time() - custom_endpoint.endpoint_timeout) > start:
                return SyftError(
                    message=(
                        f"Function timed out in {custom_endpoint.endpoint_timeout} seconds. "
                        + f"Get the Job with id: {job_id} to check results."
                    )
                )

        if job.status == JobStatus.COMPLETED:
            return job.result
        else:
            return SyftError(message="Function failed to complete.")

    @service_method(
        path="api.get_public_context", name="get_public_context", roles=ADMIN_ROLE_LEVEL
    )
    def get_public_context(
        self, context: AuthedServiceContext, path: str
    ) -> dict[str, Any] | SyftError:
        """Get specific public api context."""
        custom_endpoint = self.get_code(
            context=context,
            endpoint_path=path,
        )
        if isinstance(custom_endpoint, SyftError):
            return custom_endpoint

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
    ) -> dict[str, Any] | SyftError:
        """Get specific private api context."""
        custom_endpoint = self.get_code(
            context=context,
            endpoint_path=path,
        )
        if isinstance(custom_endpoint, SyftError):
            return custom_endpoint

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
    ) -> list[TwinAPIEndpoint] | SyftError:
        """Get all API endpoints."""
        result = self.stash.get_all(context.credentials)
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

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

        exec_result = custom_endpoint.exec(context, *args, **kwargs)

        if isinstance(exec_result, SyftError):
            return Ok(exec_result)

        action_obj = ActionObject.from_obj(exec_result)
        action_service = cast(ActionService, context.server.get_service(ActionService))
        try:
            result = action_service.set_result_to_store(
                context=context,
                result_action_object=action_obj,
                has_result_read_permission=True,
            )
            if result.is_err():
                return SyftError(
                    message=f"Failed to set result to store: {result.err()}"
                )

            return Ok(result.ok())
        except Exception as e:
            # stdlib
            import traceback

            return Err(value=f"Failed to run. {e}, {traceback.format_exc()}")

    @service_method(path="api.call_public", name="call_public", roles=GUEST_ROLE_LEVEL)
    def call_public(
        self,
        context: AuthedServiceContext,
        path: str,
        *args: Any,
        **kwargs: Any,
    ) -> ActionObject | SyftError:
        """Call a Custom API Method in public mode"""
        custom_endpoint = self.get_code(
            context=context,
            endpoint_path=path,
        )
        if isinstance(custom_endpoint, SyftError):
            return custom_endpoint
        exec_result = custom_endpoint.exec_mock_function(context, *args, **kwargs)

        if isinstance(exec_result, SyftError):
            return Ok(exec_result)

        action_obj = ActionObject.from_obj(exec_result)
        action_service = cast(ActionService, context.server.get_service(ActionService))
        try:
            result = action_service.set_result_to_store(
                context=context,
                result_action_object=action_obj,
                has_result_read_permission=True,
            )
            if result.is_err():
                return SyftError(
                    message=f"Failed to set result to store: {result.err()}"
                )

            return Ok(result.ok())
        except Exception as e:
            # stdlib
            import traceback

            return Err(value=f"Failed to run. {e}, {traceback.format_exc()}")

    @service_method(
        path="api.call_private", name="call_private", roles=GUEST_ROLE_LEVEL
    )
    def call_private(
        self,
        context: AuthedServiceContext,
        path: str,
        *args: Any,
        **kwargs: Any,
    ) -> ActionObject | SyftError:
        """Call a Custom API Method in private mode"""
        custom_endpoint = self.get_code(
            context=context,
            endpoint_path=path,
        )
        if isinstance(custom_endpoint, SyftError):
            return custom_endpoint

        exec_result = custom_endpoint.exec_private_function(context, *args, **kwargs)

        if isinstance(exec_result, SyftError):
            return Ok(exec_result)

        action_obj = ActionObject.from_obj(exec_result)

        action_service = cast(ActionService, context.server.get_service(ActionService))
        try:
            result = action_service.set_result_to_store(
                context=context, result_action_object=action_obj
            )
            if result.is_err():
                return SyftError(
                    message=f"Failed to set result to store: {result.err()}"
                )

            return Ok(result.ok())

        except Exception as e:
            # stdlib
            import traceback

            return Err(value=f"Failed to run. {e}, {traceback.format_exc()}")

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

    # ==== The methods below aren't meant to be called directly by the user, but
    # rather by the server context. ===
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
        admin_key = context.server.get_service("userservice").admin_verify_key()
        result = self.stash.get_by_uid(admin_key, uid)
        if result.is_err():
            return SyftError(message=result.err())
        return result.ok()

    def get_endpoints(
        self, context: AuthedServiceContext
    ) -> list[TwinAPIEndpoint] | SyftError:
        # TODO: Add ability to specify which roles see which endpoints
        # for now skip auth
        results = self.stash.get_all(context.server.verify_key)
        if results.is_ok():
            return results.ok()
        return SyftError(messages="Unable to get CustomAPIEndpoint")

    def get_code(
        self, context: AuthedServiceContext, endpoint_path: str
    ) -> TwinAPIEndpoint | SyftError:
        result = self.stash.get_by_path(context.server.verify_key, path=endpoint_path)
        if result.is_err():
            return SyftError(
                message=f"CustomAPIEndpoint: {endpoint_path} does not exist"
            )

        if result.is_ok():
            return result.ok()

        return SyftError(message=f"Unable to get {endpoint_path} CustomAPIEndpoint")


TYPE_TO_SERVICE[TwinAPIEndpoint] = APIService
