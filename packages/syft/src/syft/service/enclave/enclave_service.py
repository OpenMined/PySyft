# stdlib
from typing import Any
from typing import Dict
from typing import Union

# relative
from ...client.api import NodeView
from ...enclave.enclave_client import AzureEnclaveClient
from ...enclave.enclave_client import AzureEnclaveMetadata
from ...serde.serializable import serializable
from ...service.response import SyftError
from ...service.response import SyftSuccess
from ...service.user.user_roles import GUEST_ROLE_LEVEL
from ...store.document_store import DocumentStore
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.uid import UID
from ..code.user_code_service import UserCode
from ..code.user_code_service import UserCodeStatus
from ..context import AuthedServiceContext
from ..context import ChangeContext
from ..service import AbstractService
from ..service import service_method


# TODO: 🟡 Duplication of PyPrimitive Dict
# This is emulated since the action store curently accepts  only SyftObject types
@serializable()
class DictObject(SyftObject):
    # version
    __canonical_name__ = "Dict"
    __version__ = SYFT_OBJECT_VERSION_1

    base_dict: Dict[Any, Any] = {}

    # serde / storage rules
    __attr_searchable__ = []
    __attr_unique__ = ["id"]

    def __repr__(self) -> str:
        return self.base_dict.__repr__()


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

        status_update = user_code.status.mutate(
            value=UserCodeStatus.EXECUTE,
            node_name=node_name,
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

        if not action_service.exists(context=context, obj_id=user_code_id):
            dict_object = DictObject(id=user_code_id)
            dict_object.base_dict[str(context.credentials)] = inputs
            # TODO: Instead of using the action store, modify to
            # use the action service directly to store objects
            action_service.store.set(
                uid=user_code_id,
                credentials=context.node.verify_key,
                syft_object=dict_object,
                has_result_read_permission=True,
            )

        else:
            res = action_service.store.get(
                uid=user_code_id, credentials=context.node.verify_key
            )
            if res.is_ok():
                dict_object = res.ok()
                dict_object.base_dict[str(context.credentials)] = inputs
                action_service.store.set(
                    uid=user_code_id,
                    credentials=context.node.verify_key,
                    syft_object=dict_object,
                )
            else:
                return SyftError(
                    message=f"Error while fetching the object on Enclave: {res.err()}"
                )

        return SyftSuccess(message="Enclave Code Status Updated Successfully")


# Checks if the given user code would  propogate value to enclave on acceptance
def check_enclave_transfer(
    user_code: UserCode, value: UserCodeStatus, context: ChangeContext
):
    if (
        type(user_code.enclave_metadata).__name__ == "OblvMetadata"
        and value == UserCodeStatus.EXECUTE
    ):
        # relative
        from ...external import OBLV

        if OBLV:
            # relative
            from ...external.oblv.oblv_service import OblvService
        else:
            return SyftError(
                message="Oblivious is not enabled."
                "To enable oblivious package, set sy.enable_external_lib('oblv') "
                "on the client side"
                "Or add --oblv when launching by hagrid"
            )

        method = context.node.get_service_method(OblvService.get_api_for)

        api = method(
            user_code.enclave_metadata,
            context.node.signing_key,
            worker_name=context.node.name,
        )
        # send data of the current node to enclave
        user_node_view = NodeView(
            node_name=context.node.name, verify_key=context.node.signing_key.verify_key
        )
        inputs = user_code.input_policy.inputs[user_node_view]
        action_service = context.node.get_service("actionservice")
        for var_name, uid in inputs.items():
            action_object = action_service.store.get(
                uid=uid, credentials=context.node.signing_key.verify_key
            )
            if action_object.is_err():
                return SyftError(message=action_object.err())
            inputs[var_name] = action_object.ok()

        res = api.services.oblv.send_user_code_inputs_to_enclave(
            user_code_id=user_code.id, inputs=inputs, node_name=context.node.name
        )

        return res

    elif (
        isinstance(user_code.enclave_metadata, AzureEnclaveMetadata)
        and value == UserCodeStatus.EXECUTE
    ):
        # TODO 🟣 Restructure url it work for local mode host.docker.internal
        azure_enclave_client = AzureEnclaveClient.from_enclave_metadata(
            enclave_metadata=user_code.enclave_metadata,
            signing_key=context.node.signing_key,
        )

        # send data of the current node to enclave
        user_node_view = NodeView(
            node_name=context.node.name, verify_key=context.node.signing_key.verify_key
        )
        inputs = user_code.input_policy.inputs[user_node_view]
        action_service = context.node.get_service("actionservice")
        for var_name, uid in inputs.items():
            action_object = action_service.store.get(
                uid=uid, credentials=context.node.signing_key.verify_key
            )
            if action_object.is_err():
                return SyftError(message=action_object.err())
            inputs[var_name] = action_object.ok()

        res = (
            azure_enclave_client.api.services.enclave.send_user_code_inputs_to_enclave(
                user_code_id=user_code.id, inputs=inputs, node_name=context.node.name
            )
        )

        return res
    else:
        return SyftSuccess(message="Current Request does not require Enclave Transfer")
