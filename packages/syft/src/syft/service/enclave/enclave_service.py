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
from ..code.user_code import UserCode
from ..code.user_code import UserCodeStatus
from ..context import AuthedServiceContext
from ..context import ChangeContext
from ..network.routes import route_to_connection
from ..policy.policy import InputPolicy
from ..service import AbstractService
from ..service import service_method
from .enclave import Enclave


@instrument
@serializable()
class EnclaveStash(BaseUIDStoreStash):
    object_type = Enclave
    settings: PartitionSettings = PartitionSettings(
        name=Enclave.__canonical_name__, object_type=Enclave
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
        self, context: AuthedServiceContext, name: str, route: NodeRouteType
    ) -> SyftSuccess | SyftError:
        """Add an Enclave to the network."""
        enclave = Enclave.create(route=route)
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
    def get_all(self, context: AuthedServiceContext) -> list[Enclave] | SyftError:
        """Add an Enclave to the network."""
        result = self.stash.get_all(context.credentials)
        if result.is_ok():
            enclaves = result.ok()
            return enclaves
        return SyftError(message=result.err())

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
