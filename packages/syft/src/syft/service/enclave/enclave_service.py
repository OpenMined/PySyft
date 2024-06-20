# stdlib
import itertools
from typing import Any

# relative
from ...client.enclave_client import EnclaveClient
from ...client.enclave_client import EnclaveMetadata
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
from ..context import ChangeContext
from ..network.routes import route_to_connection
from ..policy.policy import InputPolicy
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


# TODO 🟣 Created a generic Enclave Service
# Currently it mainly works only for Azure
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
        self, context: AuthedServiceContext, service_func_name: str
    ) -> SyftSuccess | SyftError:
        """Request an Enclave for running a project."""
        if not context.node or not context.node.signing_key:
            return SyftError(message=f"{type(context)} has no node")

        code_service = context.node.get_service("usercodeservice")
        code: UserCode = code_service.get_by_service_name(
            context=context, service_func_name=service_func_name
        )[-1]  # TODO also match code hash and check status to get the actual code
        if not code.deployment_policy_init_kwargs:
            return SyftError(
                message=f"Code '{service_func_name}' does not have a deployment policy."
            )
        provider = code.deployment_policy_init_kwargs.get("provider")
        if not isinstance(provider, EnclaveInstance):
            return SyftError(
                message=f"Code '{service_func_name}' does not have an Enclave deployment provider."
            )
        if context.node.id != provider.syft_node_location:
            return SyftError(
                message=f"The enclave '{provider.name}' does not belong to the current domain '{context.node.name}'."
            )
        enclave_client = provider.get_client(verify_key=context.node.verify_key)
        result = enclave_client.services.enclave.setup_enclave_for_code_execution(
            code=code
        )
        return result

    @service_method(
        path="enclave.setup_enclave_for_code_execution",
        name="setup_enclave_for_code_execution",
        roles=DATA_SCIENTIST_ROLE_LEVEL,  # TODO 🟣 Only an enclave's owner domain node should call this
    )
    def setup_enclave_for_code_execution(
        self, context: AuthedServiceContext, code: UserCode | SubmitUserCode
    ) -> SyftSuccess | SyftError:
        if not context.node or not context.node.signing_key:
            return SyftError(message=f"{type(context)} has no node")

        # TODO add queuing mechanism

        if isinstance(code, UserCode):
            code = SubmitUserCode(
                code=code.raw_code,
                func_name=code.service_func_name,
                signature=code.signature,
                input_policy_type=code.input_policy_type,
                input_policy_init_kwargs=code.input_policy_init_kwargs,
                output_policy_type=code.output_policy_type,
                output_policy_init_kwargs=code.output_policy_init_kwargs,
                deployment_policy_type=code.deployment_policy_type,
                deployment_policy_init_kwargs=code.deployment_policy_init_kwargs,
                input_kwargs=code.input_kwargs,
                worker_pool_name=code.worker_pool_name,
            )

        # TODO 🟣 set up user accounts for each domain for transferring assets

        result = context.node.get_service("usercodeservice").submit(context, code)
        return result

    @service_method(
        path="enclave.request_assets_transfer_to_enclave",
        name="request_assets_transfer_to_enclave",
        roles=DATA_SCIENTIST_ROLE_LEVEL,  # TODO 🟣 update this
    )
    def request_assets_transfer_to_enclave(
        self, context: AuthedServiceContext, service_func_name: str
    ) -> SyftSuccess | SyftError:
        if not context.node or not context.node.signing_key:
            return SyftError(message=f"{type(context)} has no node")

        # Get the code
        code_service = context.node.get_service("usercodeservice")
        code: UserCode = code_service.get_by_service_name(
            context=context, service_func_name=service_func_name
        )[-1]  # TODO also match code hash and check status to get the actual code
        if code.input_policy_init_kwargs is None:
            return SyftSuccess(message="No assets to transfer")

        # Get all asset action ids for the current node
        asset_action_ids_nested = [
            assets.values()
            for node_identity, assets in code.input_policy_init_kwargs.items()
            if node_identity.node_name == context.node.name
        ]
        asset_action_ids = tuple(itertools.chain.from_iterable(asset_action_ids_nested))
        root_context = AuthedServiceContext(
            node=context.node, credentials=context.node.verify_key
        )
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
        if not code.deployment_policy_init_kwargs:
            return SyftError(
                message=f"Code '{service_func_name}' does not have a deployment policy."
            )
        provider = code.deployment_policy_init_kwargs.get("provider")
        if not isinstance(provider, EnclaveInstance):
            return SyftError(
                message=f"Code '{service_func_name}' does not have an Enclave deployment provider."
            )
        enclave_client = provider.get_client(verify_key=context.node.verify_key)

        # Upload the assets to the enclave
        result = enclave_client.api.services.enclave.upload_input_data_for_code(
            service_func_name=service_func_name, action_objects=action_objects
        )
        if isinstance(result, SyftError):
            return result

        return SyftSuccess(
            message=f"Assets transferred from Domain '{context.node.name}' to Enclave '{enclave_client.name}'"
        )

    @service_method(
        path="enclave.upload_input_data_for_code",
        name="upload_input_data_for_code",
        roles=DATA_SCIENTIST_ROLE_LEVEL,  # TODO 🟣 update this
    )
    def upload_input_data_for_code(
        self,
        context: AuthedServiceContext,
        service_func_name: str,
        action_objects: list[ActionObject] | list[TwinObject],
    ) -> SyftSuccess | SyftError:
        if not context.node or not context.node.signing_key:
            return SyftError(message=f"{type(context)} has no node")

        code_service = context.node.get_service("usercodeservice")
        action_service = context.node.get_service("actionservice")

        # Get the code
        code: UserCode = code_service.get_by_service_name(
            context=context, service_func_name=service_func_name
        )[-1]  # TODO also match code hash to get the actual code

        init_kwargs = code.input_policy_init_kwargs
        if not code or not init_kwargs:
            return SyftError(message="No assets to transfer")

        # TODO fetch the uploader node id from context once user accounts are set up for each domain
        uploader_node_id = action_objects[0].syft_node_uid

        # TODO check if the uploader node is allowed to upload assets for the given code
        # TODO only allow uploading action objects present in the input policy
        for action_object in action_objects:
            action_object.syft_node_uid = context.node.id
            action_object.syft_action_data_node_id = context.node.id
            result = action_service.set(context=context, action_object=action_object)
            if result.is_err():
                # TODO 🟣 Rollback previously uploaded assets if any error occurs
                return result

        # Let's approve the code
        kwargs_for_uploading_node = {
            name: action_id
            for node, assets in init_kwargs.items()
            for name, action_id in assets.items()
            if node.node_id == uploader_node_id
        }
        all_assets_uploaded_for_current_node = all(
            context.node.get_service("actionservice").exists(context, obj_id)
            for obj_id in kwargs_for_uploading_node.values()
        )
        if all_assets_uploaded_for_current_node:
            root_context = context.as_root_context()
            status = code.get_status(root_context)
            status.status_dict = {
                k: (
                    (
                        UserCodeStatus.APPROVED,
                        "All dependent assets uploaded by this domain node.",
                    )
                    if k.node_id == uploader_node_id
                    else v
                )
                for k, v in status.status_dict.items()
            }
            status_link = code.status_link
            if not status_link:
                return SyftError(
                    message=f"Code '{service_func_name}' does not have a status link."
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
        roles=DATA_SCIENTIST_ROLE_LEVEL,  # TODO 🟣 update this
    )
    def request_execution(
        self, context: AuthedServiceContext, service_func_name: str
    ) -> Any:
        if not context.node or not context.node.signing_key:
            return SyftError(message=f"{type(context)} has no node")

        code_service = context.node.get_service("usercodeservice")
        code: UserCode = code_service.get_by_service_name(
            context=context, service_func_name=service_func_name
        )[-1]

        if not code.deployment_policy_init_kwargs:
            return SyftError(
                message=f"Code '{service_func_name}' does not have a deployment policy."
            )
        provider = code.deployment_policy_init_kwargs.get("provider")
        if not isinstance(provider, EnclaveInstance):
            return SyftError(
                message=f"Code '{service_func_name}' does not have an Enclave deployment provider."
            )

        enclave_client = provider.get_client(verify_key=context.node.verify_key)
        result = enclave_client.api.services.enclave.execute_code(
            service_func_name=service_func_name
        )
        return result

    @service_method(
        path="enclave.execute_code",
        name="execute_code",
        roles=DATA_SCIENTIST_ROLE_LEVEL,  # TODO 🟣 update this
    )
    def execute_code(
        self, context: AuthedServiceContext, service_func_name: str
    ) -> Any:
        if not context.node or not context.node.signing_key:
            return SyftError(message=f"{type(context)} has no node")

        code_service = context.node.get_service("usercodeservice")
        code: UserCode = code_service.get_by_service_name(
            context=context, service_func_name=service_func_name
        )[-1]  # TODO also match code hash to get the actual code

        init_kwargs = (
            code.input_policy_init_kwargs.values()
            if code.input_policy_init_kwargs is not None
            else []
        )
        kwargs = {k: v for d in init_kwargs for k, v in d.items()}

        # TODO only allow execution for domain nodes in output_policy.share_result_with list
        admin_context = context.as_root_context()
        execution_result = code_service.call(
            context=admin_context, uid=code.id, **kwargs
        ).syft_action_data
        result = get_encrypted_result(context, execution_result)
        return result

    @service_method(
        path="enclave.send_user_code_inputs_to_enclave",
        name="send_user_code_inputs_to_enclave",
        roles=GUEST_ROLE_LEVEL,
    )
    def send_user_code_inputs_to_enclave(
        self,
        context: AuthedServiceContext,
        user_code_id: UID,
        inputs: dict,
        node_name: str,
        node_id: UID,
    ) -> SyftSuccess | SyftError:
        if not context.node or not context.node.signing_key:
            return SyftError(message=f"{type(context)} has no node")

        root_context = AuthedServiceContext(
            credentials=context.node.verify_key, node=context.node
        )

        user_code_service = context.node.get_service("usercodeservice")
        action_service = context.node.get_service("actionservice")
        user_code = user_code_service.get_by_uid(context=root_context, uid=user_code_id)
        if isinstance(user_code, SyftError):
            return user_code

        reason: str = context.extra_kwargs.get("reason", "")
        status_update = user_code.get_status(root_context).mutate(
            value=(UserCodeStatus.APPROVED, reason),
            node_name=node_name,
            node_id=node_id,
            verify_key=context.credentials,
        )
        if isinstance(status_update, SyftError):
            return status_update

        res = user_code.status_link.update_with_context(root_context, status_update)
        if isinstance(res, SyftError):
            return res

        root_context = context.as_root_context()
        if not action_service.exists(context=context, obj_id=user_code_id):
            dict_object = ActionObject.from_obj({})
            dict_object.id = user_code_id
            dict_object[str(context.credentials)] = inputs
            root_context.extra_kwargs = {"has_result_read_permission": True}
            # TODO: Instead of using the action store, modify to
            # use the action service directly to store objects
            action_service.set(root_context, dict_object)

        else:
            res = action_service.get(uid=user_code_id, context=root_context)
            if res.is_ok():
                dict_object = res.ok()
                dict_object[str(context.credentials)] = inputs
                action_service.set(root_context, dict_object)
            else:
                return SyftError(
                    message=f"Error while fetching the object on Enclave: {res.err()}"
                )

        return SyftSuccess(message="Enclave Code Status Updated Successfully")


# Checks if the given user code would propogate value to enclave on acceptance
def propagate_inputs_to_enclave(
    user_code: UserCode, context: ChangeContext
) -> SyftSuccess | SyftError:
    if isinstance(user_code.enclave_metadata, EnclaveMetadata):
        # TODO 🟣 Restructure url it work for local mode host.docker.internal

        connection = route_to_connection(user_code.enclave_metadata.route)
        enclave_client = EnclaveClient(
            connection=connection,
            credentials=context.node.signing_key,
        )

        send_method = (
            enclave_client.api.services.enclave.send_user_code_inputs_to_enclave
        )

    else:
        return SyftSuccess(message="Current Request does not require Enclave Transfer")

    input_policy: InputPolicy | None = user_code.get_input_policy(
        context.to_service_ctx()
    )
    if input_policy is None:
        return SyftError(message=f"{user_code}'s input policy is None")
    inputs = input_policy._inputs_for_context(context)
    if isinstance(inputs, SyftError):
        return inputs

    # Save inputs to blob store
    for var_name, var_value in inputs.items():
        if isinstance(var_value, ActionObject | TwinObject):
            # Set the obj location to enclave
            var_value._set_obj_location_(
                enclave_client.api.node_uid,
                enclave_client.verify_key,
            )
            var_value._save_to_blob_storage()

            inputs[var_name] = var_value

    # send data of the current node to enclave
    res = send_method(
        user_code_id=user_code.id,
        inputs=inputs,
        node_name=context.node.name,
        node_id=context.node.id,
    )
    return res


def get_encrypted_result(context: AuthedServiceContext, result: Any) -> Any:
    # TODO 🟣 Encrypt the result before sending it back to the user
    return result
