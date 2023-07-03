# stdlib
from typing import Any
from typing import Dict
from typing import Optional

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ...serde.serializable import serializable
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
from ..service import SyftError
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
    service_name: Optional[str]

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
    ) -> Result[Ok, Err]:
        if not context.node or not context.node.signing_key:
            return Err(f"{type(context)} has no node")

        user_code_service = context.node.get_service("usercodeservice")
        action_service = context.node.get_service("actionservice")
        user_code = user_code_service.stash.get_by_uid(
            context.node.signing_key.verify_key, uid=user_code_id
        )
        if user_code.is_err():
            return SyftError(
                message=f"Unable to find {user_code_id} in {type(user_code_service)}"
            )
        user_code = user_code.ok()

        res = user_code.status.mutate(
            value=UserCodeStatus.EXECUTE,
            node_name=node_name,
            node_id=node_id,
            verify_key=context.credentials,
        )
        if res.is_err():
            return res
        user_code.status = res.ok()

        root_context = AuthedServiceContext(credentials=context.node.verify_key)
        user_code_update = user_code_service.update_code_state(
            context=root_context, code_item=user_code
        )
        if isinstance(user_code_update, SyftError):
            return user_code_update

        if not action_service.exists(context=context, obj_id=user_code_id):
            dict_object = DictObject(id=user_code_id)
            dict_object.base_dict[str(context.credentials)] = inputs
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
                return res

        return Ok(Ok(True))


# Checks if the given user code would  propogate value to enclave on acceptance
def propagate_inputs_to_enclave(user_code: UserCode, context: ChangeContext):
    # relative
    from ...enclave.enclave_client import AzureEnclaveClient
    from ...enclave.enclave_client import AzureEnclaveMetadata
    from ...external.oblv.deployment_client import OblvMetadata
    from ...external.oblv.oblv_service import OblvService

    if isinstance(user_code.enclave_metadata, OblvMetadata):
        method = context.node.get_service_method(OblvService.get_api_for)

        api = method(
            user_code.enclave_metadata,
            context.node.signing_key,
        )
        send_method = api.services.oblv.send_user_code_inputs_to_enclave

        inputs = user_code.input_policy._inputs_for_context(context)
        if inputs.is_err():
            return inputs
        else:
            inputs = inputs.ok()
    elif isinstance(user_code.enclave_metadata, AzureEnclaveMetadata):
        # TODO 🟣 Restructure url it work for local mode host.docker.internal

        azure_enclave_client = AzureEnclaveClient.from_enclave_metadata(
            enclave_metadata=user_code.enclave_metadata,
            signing_key=context.node.signing_key,
        )

        # send data of the current node to enclave
        inputs = user_code.input_policy._inputs_for_context(context)
        if inputs.is_err():
            return inputs
        else:
            inputs = inputs.ok()
        send_method = (
            azure_enclave_client.api.services.enclave.send_user_code_inputs_to_enclave
        )
    else:
        return Ok()

    res = send_method(
        user_code_id=user_code.id,
        inputs=inputs,
        node_name=context.node.name,
        node_id=context.node.id,
    )
    return res
