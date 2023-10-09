# stdlib
from typing import Dict
from typing import Union

# relative
from ...client.enclave_client import EnclaveClient
from ...client.enclave_client import EnclaveMetadata
from ...serde.serializable import serializable
from ...service.response import SyftError
from ...service.response import SyftSuccess
from ...service.user.user_roles import GUEST_ROLE_LEVEL
from ...store.document_store import DocumentStore
from ...types.twin_object import TwinObject
from ...types.uid import UID
from ..action.action_object import ActionObject
from ..code.user_code_service import UserCode
from ..code.user_code_service import UserCodeStatus
from ..context import AuthedServiceContext
from ..context import ChangeContext
from ..network.routes import route_to_connection
from ..service import AbstractService
from ..service import service_method


# TODO 🟣 Created a generic Enclave Service
# Currently it mainly works only for Azure
@serializable()
class EnclaveService(AbstractService):
    store: DocumentStore

    def __init__(self, store: DocumentStore) -> None:
        self.store = store

    @service_method(
        path="enclave.send_user_code_inputs_to_enclave",
        name="send_user_code_inputs_to_enclave",
        roles=GUEST_ROLE_LEVEL,
    )
    def send_user_code_inputs_to_enclave(
        self,
        context: AuthedServiceContext,
        user_code_id: UID,
        inputs: Dict,
        node_name: str,
        node_id: UID,
    ) -> Union[SyftSuccess, SyftError]:
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
        status_update = user_code.status.mutate(
            value=(UserCodeStatus.APPROVED, reason),
            node_name=node_name,
            node_id=node_id,
            verify_key=context.credentials,
        )
        if isinstance(status_update, SyftError):
            return status_update

        user_code.status = status_update

        user_code_update = user_code_service.update_code_state(
            context=root_context, code_item=user_code
        )
        if isinstance(user_code_update, SyftError):
            return user_code_update

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


def get_oblv_service():
    # relative
    from ...external import OBLV

    if OBLV:
        # relative
        from ...external.oblv.oblv_service import OblvService

        return OblvService
    else:
        return SyftError(
            message="Oblivious is not enabled."
            "To enable oblivious package, set sy.enable_external_lib('oblv') "
            "on the client side"
            "Or add --oblv when launching by hagrid"
        )


# Checks if the given user code would  propogate value to enclave on acceptance
def propagate_inputs_to_enclave(user_code: UserCode, context: ChangeContext):
    # Temporarily disable Oblivious Enclave
    # from ...external.oblv.deployment_client import OblvMetadata

    # if isinstance(user_code.enclave_metadata, OblvMetadata):
    #     # relative
    #     oblv_service_class = get_oblv_service()
    #     if isinstance(oblv_service_class, SyftError):
    #         return oblv_service_class
    #     method = context.node.get_service_method(oblv_service_class.get_api_for)

    #     api = method(
    #         user_code.enclave_metadata,
    #         context.node.signing_key,
    #         worker_name=context.node.name,
    #     )
    #     send_method = api.services.oblv.send_user_code_inputs_to_enclave

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

    inputs = user_code.input_policy._inputs_for_context(context)
    if isinstance(inputs, SyftError):
        return inputs

    # Save inputs to blob store
    for var_name, var_value in inputs.items():
        if isinstance(var_value, (ActionObject, TwinObject)):
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
