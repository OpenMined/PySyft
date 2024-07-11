# stdlib
import itertools
from typing import Any

# relative
from ...serde.serializable import serializable
from ...service.action.action_permissions import ActionObjectPermission
from ...service.action.action_permissions import ActionPermission
from ...service.network.routes import NodeRouteType
from ...service.response import SyftError
from ...service.response import SyftSuccess
from ...service.user.user_roles import ADMIN_ROLE_LEVEL
from ...service.user.user_roles import DATA_SCIENTIST_ROLE_LEVEL
from ...store.document_store import DocumentStore
from ...types.uid import UID
from ..code.user_code import UserCode
from ..context import AuthedServiceContext
from ..model.model import ModelRef
from ..service import AbstractService
from ..service import service_method
from .enclave import EnclaveInstance
from .enclave_stash import EnclaveInstanceStash


@serializable()
class DomainEnclaveService(AbstractService):
    """Contains service methods for Domain -> Enclave communication."""

    store: DocumentStore
    stash: EnclaveInstanceStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = EnclaveInstanceStash(store=store)

    @service_method(
        path="enclave.add",
        name="add",
        roles=ADMIN_ROLE_LEVEL,
    )
    def add(
        self, context: AuthedServiceContext, route: NodeRouteType
    ) -> SyftSuccess | SyftError:
        """Add an Enclave to the network."""
        enclave = EnclaveInstance(route=route)
        result = self.stash.set(
            credentials=context.credentials,
            obj=enclave,
            add_permissions=[
                ActionObjectPermission(
                    uid=enclave.id, permission=ActionPermission.ALL_READ
                )
            ],
        )
        if result.is_err():
            return SyftError(message=str(result.err()))
        return SyftSuccess(
            message=f"Enclave '{enclave.name}' added to '{context.node.name}' on '{route}'."
        )

    @service_method(
        path="enclave.get_all",
        name="get_all",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get_all(
        self, context: AuthedServiceContext
    ) -> list[EnclaveInstance] | SyftError:
        """Add an Enclave to the network."""
        result = self.stash.get_all(context.credentials)
        if result.is_ok():
            enclaves = result.ok()
            return enclaves
        return SyftError(message=result.err())

    @service_method(
        path="enclave.request_enclave",
        name="request_enclave",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def request_enclave(
        self, context: AuthedServiceContext, user_code_id: UID
    ) -> SyftSuccess | SyftError:
        """Request an Enclave for running a project."""
        if not context.node or not context.node.signing_key:
            return SyftError(message=f"{type(context)} has no node")

        code_service = context.node.get_service("usercodeservice")
        code: UserCode = code_service.get_by_uid(context=context, uid=user_code_id)
        status = code.get_status(context)
        if not status.approved:
            return SyftError(
                message=f"Status for code '{code.service_func_name}' is not Approved."
            )
        if not code.runtime_policy_init_kwargs:
            return SyftError(
                message=f"Code '{code.service_func_name}' does not have a deployment policy."
            )
        provider = code.runtime_policy_init_kwargs.get("provider")
        if not isinstance(provider, EnclaveInstance):
            return SyftError(
                message=f"Code '{code.service_func_name}' does not have an Enclave deployment provider."
            )
        if context.node.id != provider.syft_node_location:
            return SyftError(
                message=f"The enclave '{provider.name}' does not belong to the current domain '{context.node.name}'."
            )

        current_node_credentials = context.node.signing_key
        enclave_client = provider.get_client(credentials=current_node_credentials)

        result = enclave_client.api.services.enclave.setup_enclave(code=code)
        return result

    @service_method(
        path="enclave.request_assets_upload",
        name="request_assets_upload",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def request_assets_upload(
        self, context: AuthedServiceContext, user_code_id: UID
    ) -> SyftSuccess | SyftError:
        if not context.node or not context.node.signing_key:
            return SyftError(message=f"{type(context)} has no node")

        root_context = context.as_root_context()

        # Get the code
        code_service = context.node.get_service("usercodeservice")
        code: UserCode = code_service.get_by_uid(context=context, uid=user_code_id)

        status = code.get_status(context)
        if not status.approved:
            return SyftError(
                message=f"Code '{code.service_func_name}' is not approved."
            )

        if code.input_policy_init_kwargs is None:
            return SyftSuccess(message="No assets to transfer")

        # Get all asset action ids for the current node
        asset_action_ids_nested = [
            assets.values()
            for node_identity, assets in code.input_policy_init_kwargs.items()
            if node_identity.node_id == context.node.id
        ]
        asset_action_ids = tuple(itertools.chain.from_iterable(asset_action_ids_nested))
        action_objects = [
            context.node.get_service("actionservice")
            .get(context=root_context, uid=action_id)
            .ok()
            for action_id in asset_action_ids
        ]
        # Actual data from blob storage is lazy-loaded when the `syft_action_data` property is used for the
        # first time. Let's load it now so that it can get properly transferred along with the action objects.
        for action_object in action_objects:
            # If it is ModelRef, then load all the references
            # and wrap them to the Model Ref object
            if isinstance(action_object, ModelRef):
                action_object.load_data(
                    context=context, wrap_ref_to_obj=True, unwrap_action_data=False
                )

            _ = action_object.syft_action_data

        # Get the enclave client
        if not code.runtime_policy_init_kwargs:
            return SyftError(
                message=f"Code '{code.service_func_name}' does not have a deployment policy."
            )
        provider = code.runtime_policy_init_kwargs.get("provider")
        if not isinstance(provider, EnclaveInstance):
            return SyftError(
                message=f"Code '{code.service_func_name}' does not have an Enclave deployment provider."
            )

        current_node_credentials = context.node.signing_key
        enclave_client = provider.get_client(credentials=current_node_credentials)

        # Upload the assets to the enclave
        result = enclave_client.api.services.enclave.upload_assets(
            user_code_id=user_code_id, action_objects=action_objects
        )
        if isinstance(result, SyftError):
            return result

        return SyftSuccess(
            message=f"Assets transferred from Domain '{context.node.name}' to Enclave '{enclave_client.name}'"
        )

    @service_method(
        path="enclave.request_code_execution",
        name="request_code_execution",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def request_code_execution(
        self, context: AuthedServiceContext, user_code_id: UID
    ) -> Any:
        if not context.node or not context.node.signing_key:
            return SyftError(message=f"{type(context)} has no node")

        code_service = context.node.get_service("usercodeservice")
        code: UserCode = code_service.get_by_uid(context=context, uid=user_code_id)

        status = code.get_status(context)
        if not status.approved:
            return SyftError(
                message=f"Code '{code.service_func_name}' is not approved."
            )

        if not code.runtime_policy_init_kwargs:
            return SyftError(
                message=f"Code '{code.service_func_name}' does not have a deployment policy."
            )
        provider = code.runtime_policy_init_kwargs.get("provider")
        if not isinstance(provider, EnclaveInstance):
            return SyftError(
                message=f"Code '{code.service_func_name}' does not have an Enclave deployment provider."
            )

        current_node_credentials = context.node.signing_key
        enclave_client = provider.get_client(credentials=current_node_credentials)

        result = enclave_client.api.services.enclave.execute_code(
            user_code_id=user_code_id
        )
        return result
