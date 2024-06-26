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
from ...service.user.user_roles import GUEST_ROLE_LEVEL
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionSettings
from ...types.twin_object import TwinObject
from ...types.uid import UID
from ...util.telemetry import instrument
from ..action.action_object import ActionObject
from ..code.user_code import SubmitUserCode
from ..code.user_code import UserCode
from ..code.user_code import UserCodeStatus
from ..context import AuthedServiceContext
from ..service import AbstractService
from ..service import service_method
from .enclave import EnclaveInstance


@instrument
@serializable()
class EnclaveStash(BaseUIDStoreStash):
    object_type = EnclaveInstance
    settings: PartitionSettings = PartitionSettings(
        name=EnclaveInstance.__canonical_name__, object_type=EnclaveInstance
    )


@serializable()
class EnclaveService(AbstractService):
    store: DocumentStore
    stash: EnclaveStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = EnclaveStash(store=store)

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
        path="enclave.request_enclave_for_code_execution",
        name="request_enclave_for_code_execution",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def request_enclave_for_code_execution(
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
                message=f"Code '{code.service_func_name}' does not have a runtime policy."
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

        result = enclave_client.api.services.enclave.setup_enclave_for_code_execution(
            code=code
        )
        return result

    @service_method(
        path="enclave.setup_enclave_for_code_execution",
        name="setup_enclave_for_code_execution",
        roles=GUEST_ROLE_LEVEL,  # TODO 🟣 Only an enclave's owner domain node should call this
    )
    def setup_enclave_for_code_execution(
        self, context: AuthedServiceContext, code: UserCode | SubmitUserCode
    ) -> SyftSuccess | SyftError:
        if not context.node or not context.node.signing_key:
            return SyftError(message=f"{type(context)} has no node")

        root_context = context.as_root_context()

        # TODO add queuing mechanism

        if isinstance(code, UserCode):
            code = code.to(SubmitUserCode)

        result = context.node.get_service("usercodeservice").submit(root_context, code)
        return result

    @service_method(
        path="enclave.request_assets_transfer_to_enclave",
        name="request_assets_transfer_to_enclave",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def request_assets_transfer_to_enclave(
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
            if node_identity.node_name == context.node.name
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
        [action_object.syft_action_data for action_object in action_objects]

        # Get the enclave client
        if not code.runtime_policy_init_kwargs:
            return SyftError(
                message=f"Code '{code.service_func_name}' does not have a runtime policy."
            )
        provider = code.runtime_policy_init_kwargs.get("provider")
        if not isinstance(provider, EnclaveInstance):
            return SyftError(
                message=f"Code '{code.service_func_name}' does not have an Enclave deployment provider."
            )

        current_node_credentials = context.node.signing_key
        enclave_client = provider.get_client(credentials=current_node_credentials)

        # Upload the assets to the enclave
        result = enclave_client.api.services.enclave.upload_input_data_for_code(
            user_code_id=user_code_id, action_objects=action_objects
        )
        if isinstance(result, SyftError):
            return result

        return SyftSuccess(
            message=f"Assets transferred from Domain '{context.node.name}' to Enclave '{enclave_client.name}'"
        )

    @service_method(
        path="enclave.upload_input_data_for_code",
        name="upload_input_data_for_code",
        roles=GUEST_ROLE_LEVEL,
    )
    def upload_input_data_for_code(
        self,
        context: AuthedServiceContext,
        user_code_id: UID,
        action_objects: list[ActionObject] | list[TwinObject],
    ) -> SyftSuccess | SyftError:
        if not context.node or not context.node.signing_key:
            return SyftError(message=f"{type(context)} has no node")

        root_context = context.as_root_context()

        code_service = context.node.get_service("usercodeservice")
        action_service = context.node.get_service("actionservice")

        # Get the code
        code: UserCode = code_service.get_by_uid(context=root_context, uid=user_code_id)

        init_kwargs = code.input_policy_init_kwargs
        if not code or not init_kwargs:
            return SyftError(message="No assets to transfer")

        node_identity_map = {node.verify_key: node for node in init_kwargs.keys()}
        uploading_domain_identity = node_identity_map.get(context.credentials)

        if not uploading_domain_identity:
            return SyftError(
                message="You are not allowed to upload assets for the given code"
            )

        kwargs_for_uploading_domain = init_kwargs[uploading_domain_identity]

        for action_object in action_objects:
            if action_object.id not in kwargs_for_uploading_domain.values():
                return SyftError(
                    message=f"You are not allowed to upload the asset with id '{action_object.id}'"
                )

        pending_assets_for_uploading_domain = set(kwargs_for_uploading_domain.values())
        for action_object in action_objects:
            result = action_service._set(
                root_context,
                action_object,
                ignore_detached_objs=True,
                skip_clear_cache=True,
            )
            if result.is_err():
                # TODO 🟣 Rollback previously uploaded assets if any error occurs
                return SyftError(message=result.value)
            pending_assets_for_uploading_domain.remove(action_object.id)

        # Let's approve the code
        if len(pending_assets_for_uploading_domain) == 0:
            approved_status_with_reason = (
                UserCodeStatus.APPROVED,
                "All dependent assets uploaded by this domain node.",
            )
            status = code.get_status(root_context)
            status.status_dict[uploading_domain_identity] = approved_status_with_reason
            status_link = code.status_link
            if not status_link:
                return SyftError(
                    message=f"Code '{code.service_func_name}' does not have a status link."
                )
            res = status_link.update_with_context(root_context, status)
            if isinstance(res, SyftError):
                return res

        return SyftSuccess(
            message=f"{len(action_objects)} assets uploaded successfully"
        )

    @service_method(
        path="enclave.request_execution",
        name="request_execution",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def request_execution(
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
                message=f"Code '{code.service_func_name}' does not have a runtime policy."
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

    @service_method(
        path="enclave.execute_code",
        name="execute_code",
        roles=GUEST_ROLE_LEVEL,
    )
    def execute_code(self, context: AuthedServiceContext, user_code_id: UID) -> Any:
        if not context.node or not context.node.signing_key:
            return SyftError(message=f"{type(context)} has no node")

        root_context = context.as_root_context()

        code_service = context.node.get_service("usercodeservice")
        code: UserCode = code_service.get_by_uid(context=root_context, uid=user_code_id)

        init_kwargs = (
            code.input_policy_init_kwargs.values()
            if code.input_policy_init_kwargs is not None
            else []
        )
        kwargs = {k: v for d in init_kwargs for k, v in d.items()}

        # TODO only allow execution for domain nodes in output_policy.share_result_with list
        execution_result = code_service.call(
            context=root_context, uid=code.id, **kwargs
        ).syft_action_data
        result = get_encrypted_result(context, execution_result)
        return result


def get_encrypted_result(context: AuthedServiceContext, result: Any) -> Any:
    # TODO 🟣 Encrypt the result before sending it back to the user
    return result
