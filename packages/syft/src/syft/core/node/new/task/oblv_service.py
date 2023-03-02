# stdlib
from base64 import encodebytes
from datetime import datetime
import os
import random
import subprocess  # nosec
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import cast

# third party
from oblv import OblvClient
import requests
from result import Err
from result import Ok
from result import Result

# relative
from .....core.node.common.node_table.syft_object import SYFT_OBJECT_VERSION_1
from .....core.node.common.node_table.syft_object import SyftObject
from .....oblv.constants import DOMAIN_CONNECTION_PORT
from .....oblv.constants import LOCAL_MODE
from .....oblv.constants import WORKER_MODE
from .....oblv.deployment_client import OblvMetadata
from ....common.serde.deserialize import _deserialize as deserialize
from ....common.serde.serializable import serializable
from ....common.uid import UID
from ...common.exceptions import OblvEnclaveError
from ...common.exceptions import OblvProxyConnectPCRError
from ..api import SyftAPI
from ..api import UserNodeView
from ..client import HTTPConnection
from ..client import Routes
from ..context import AuthedServiceContext
from ..context import ChangeContext
from ..credentials import SyftSigningKey
from ..document_store import DocumentStore
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import SERVICE_TO_TYPES
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from ..user_code import UserCode
from ..user_code import UserCodeStatus
from .enclave_transfer import EnclaveTransfer
from .enclave_transfer import EnclaveTransferStatus
from .enclave_transfer_stash import EnclaveTransferStash
from .oblv_keys import OblvKeys
from .oblv_keys_stash import OblvKeysStash
from .util import find_available_port

# caches the connection to Enclave using the deployment ID
OBLV_PROCESS_CACHE: Dict[str, List] = {}


# TODO: 🟡 Duplication of PyPrimitive Dict
# This is emulated since the action store curently accepts  only SyftObject types
@serializable(recursive_serde=True)
class DictObject(SyftObject):
    # version
    __canonical_name__ = "Dict"
    __version__ = SYFT_OBJECT_VERSION_1

    base_dict: Dict[Any, Any] = {}

    # serde / storage rules
    __attr_state__ = ["id", "base_dict"]

    __attr_searchable__ = []
    __attr_unique__ = ["id"]

    def __repr__(self) -> str:
        return self.base_dict.__repr__()


def connect_to_enclave(
    oblv_keys_stash: OblvKeysStash,
    oblv_client: OblvClient,
    deployment_id: str,
    connection_port: int,
) -> subprocess.Popen:
    global OBLV_PROCESS_CACHE
    if deployment_id in OBLV_PROCESS_CACHE:
        process = OBLV_PROCESS_CACHE[deployment_id][0]
        if process.poll() is None:
            return process
        # If the process has been terminated create a new connection
        del OBLV_PROCESS_CACHE[deployment_id]

    # Always create key file each time, which ensures consistency when there is key change in database
    create_keys_from_db(oblv_keys_stash)
    key_path = os.getenv("OBLV_KEY_PATH", "/app/content")
    # Temporary new key name for the new service
    key_name = os.getenv("OBLV_NEW_KEY_NAME", "new_oblv_key")
    public_file_name = key_path + "/" + key_name + "_public.der"
    private_file_name = key_path + "/" + key_name + "_private.der"

    depl = oblv_client.deployment_info(deployment_id)
    if depl.is_deleted:
        raise OblvEnclaveError(
            "User cannot connect to this deployment, as it is no longer available."
        )
    if depl.is_dev_env:
        process = subprocess.Popen(  # nosec
            [
                "/usr/local/bin/oblv",
                "connect",
                "--private-key",
                private_file_name,
                "--public-key",
                public_file_name,
                "--url",
                depl.instance.service_url,
                "--pcr0",
                depl.pcr_codes[0],
                "--pcr1",
                depl.pcr_codes[1],
                "--pcr2",
                depl.pcr_codes[2],
                "--port",
                "443",
                "--lport",
                str(connection_port),
                "--disable-pcr-check",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    else:
        process = subprocess.Popen(  # nosec
            [
                "/usr/local/bin/oblv",
                "connect",
                "--private-key",
                private_file_name,
                "--public-key",
                public_file_name,
                "--url",
                depl.instance.service_url,
                "--pcr0",
                depl.pcr_codes[0],
                "--pcr1",
                depl.pcr_codes[1],
                "--pcr2",
                depl.pcr_codes[2],
                "--port",
                "443",
                "--lport",
                str(connection_port),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    while process.poll() is None:
        log_line = process.stderr.readline().decode()
        if log_line.__contains__("Error:  Invalid PCR Values"):
            raise OblvProxyConnectPCRError()
        elif log_line.lower().__contains__("error"):
            raise OblvEnclaveError(message=log_line)
        elif log_line.__contains__("listening on"):
            break

    OBLV_PROCESS_CACHE[deployment_id] = [process, connection_port]


def make_request_to_enclave(
    oblv_keys_stash: OblvKeysStash,
    deployment_id: str,
    oblv_client: OblvClient,
    request_method: Callable,
    connection_string: str,
    connection_port: int,
    params: Optional[Dict] = None,
    files: Optional[Dict] = None,
    data: Optional[Dict] = None,
    json: Optional[Dict] = None,
):
    if not LOCAL_MODE:
        _ = connect_to_enclave(
            oblv_keys_stash=oblv_keys_stash,
            oblv_client=oblv_client,
            deployment_id=deployment_id,
            connection_port=connection_port,
        )
        req = request_method(
            connection_string,
            params=params,
            files=files,
            data=data,
            json=json,
        )

        return req
    else:
        headers = {"x-oblv-user-name": "enclave-test", "x-oblv-user-role": "domain"}

        return request_method(
            connection_string,
            headers=headers,
            params=params,
            files=files,
            data=data,
            json=json,
        )


def create_keys_from_db(oblv_keys_stash: OblvKeysStash):
    file_path = os.getenv("OBLV_KEY_PATH", "/app/content")
    # Temporary new key name for the new service
    file_name = os.getenv("OBLV_NEW_KEY_NAME", "new_oblv_key")

    keys = oblv_keys_stash.get_all()[0]

    # Creating directory if not exist
    os.makedirs(
        os.path.dirname(file_path + "/" + file_name + "_private.der"), exist_ok=True
    )
    f_private = open(file_path + "/" + file_name + "_private.der", "w+b")
    f_private.write(keys.private_key)
    f_private.close()
    f_public = open(file_path + "/" + file_name + "_public.der", "w+b")
    f_public.write(keys.public_key)
    f_public.close()


def generate_oblv_key() -> Tuple[bytes]:
    file_path = os.getenv("OBLV_KEY_PATH", "/app/content")
    # Temporary new key name for the new service
    file_name = os.getenv("OBLV_NEW_KEY_NAME", "new_oblv_key")
    result = subprocess.run(  # nosec
        [
            "/usr/local/bin/oblv",
            "keygen",
            "--key-name",
            file_name,
            "--output",
            file_path,
        ],
        capture_output=True,
    )
    if result.stderr:
        raise Err(
            subprocess.CalledProcessError(  # nosec
                returncode=result.returncode, cmd=result.args, stderr=result.stderr
            )
        )

    f_private = open(file_path + "/" + file_name + "_private.der", "rb")
    private_key = f_private.read()
    f_private.close()
    f_public = open(file_path + "/" + file_name + "_public.der", "rb")
    public_key = f_public.read()
    f_public.close()

    return (public_key, private_key)


@serializable(recursive_serde=True)
class OblvService(AbstractService):
    store: DocumentStore
    oblv_keys_stash: OblvKeysStash
    enclave_transfer_stash: EnclaveTransferStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.oblv_keys_stash = OblvKeysStash(store=store)
        self.enclave_transfer_stash = EnclaveTransferStash(store=store)

    @service_method(path="oblv.create_key", name="create_key")
    def create_key(
        self,
        context: AuthedServiceContext,
        override_existing_key: bool = False,
    ) -> Result[Ok, Err]:
        """Domain Public/Private Key pair creation"""
        # TODO 🟣 Check for permission after it is fully integrated
        public_key, private_key = generate_oblv_key()

        if override_existing_key:
            self.oblv_keys_stash.clear()
        oblv_keys = OblvKeys(public_key=public_key, private_key=private_key)

        res = self.oblv_keys_stash.set(oblv_keys)

        if res.is_ok():
            return Ok(
                "Successfully created a new public/private RSA key-pair on the domain node"
            )
        return res.err()

    @service_method(path="oblv.get_public_key", name="get_public_key")
    def get_public_key(
        self,
        context: AuthedServiceContext,
    ) -> Result[Ok, Err]:
        "Retrieves the public key present on the Domain Node."

        if len(self.oblv_keys_stash):
            oblv_keys = self.oblv_keys_stash.get_all()
            if oblv_keys.is_ok():
                oblv_keys = oblv_keys.ok()[0]
            else:
                return oblv_keys.err()

            public_key_str = (
                encodebytes(oblv_keys.public_key).decode("UTF-8").replace("\n", "")
            )
            return Ok(public_key_str)

        return Err(
            "Public Key not present for the domain node, Kindly request the admin to create a new one"
        )

    def get_api_for(
        self,
        enclave_metadata: OblvMetadata,
        signing_key: SyftSigningKey,
    ):
        deployment_id = enclave_metadata.deployment_id
        oblv_client = enclave_metadata.oblv_client
        if not LOCAL_MODE:
            if (
                deployment_id in OBLV_PROCESS_CACHE
                and OBLV_PROCESS_CACHE[deployment_id][0].poll() is None
            ):
                port = OBLV_PROCESS_CACHE[deployment_id][1]
            else:
                # randomized port staring point, to quickly find free port
                port_start = 3000 + random.randint(1, 10_000)  # nosec
                port = find_available_port(
                    host="127.0.0.1", port=port_start, search=True
                )
            connection_string = f"http://127.0.0.1:{port}"
        else:
            port = os.getenv("DOMAIN_CONNECTION_PORT", DOMAIN_CONNECTION_PORT)
            connection_string = f"http://127.0.0.1:{port}"
            if not WORKER_MODE:
                connection_string = connection_string.replace(
                    "127.0.0.1", "host.docker.internal"
                )

        req = make_request_to_enclave(
            connection_string=connection_string + Routes.ROUTE_API.value,
            deployment_id=deployment_id,
            oblv_client=oblv_client,
            oblv_keys_stash=self.oblv_keys_stash,
            request_method=requests.get,
            connection_port=port,
        )

        obj = deserialize(req.content, from_bytes=True)
        # TODO 🟣 Retrieve of signing key of user after permission  is fully integrated
        obj.signing_key = signing_key
        obj.connection = HTTPConnection(connection_string)
        return cast(SyftAPI, obj)

    @service_method(
        path="oblv.send_user_code_inputs_to_enclave",
        name="send_user_code_inputs_to_enclave",
    )
    def send_user_code_inputs_to_enclave(
        self,
        context: AuthedServiceContext,
        user_code_id: UID,
        inputs: Dict,
        node_name: str,
    ) -> Result[Ok, Err]:
        user_code_service = context.node.get_service("usercodeservice")
        action_service = context.node.get_service("actionservice")
        user_code = user_code_service.get_by_uid(context, uid=user_code_id)

        res = user_code.status.mutate(
            value=UserCodeStatus.EXECUTE,
            node_name=node_name,
            verify_key=context.credentials,
        )
        if res.is_err():
            return res
        user_code.status = res.ok()
        user_code_service.update_code_state(context=context, code_item=user_code)

        if not action_service.exists(context=context, obj_id=user_code_id):
            dict_object = DictObject(id=user_code_id)
            dict_object.base_dict[str(context.credentials)] = inputs
            action_service.store.set(
                uid=user_code_id,
                credentials=context.node.signing_key.verify_key,
                syft_object=dict_object,
            )

        else:
            res = action_service.store.get(
                uid=user_code_id, credentials=context.node.signing_key.verify_key
            )
            if res.is_ok():
                dict_object = res.ok()
                dict_object.base_dict[str(context.credentials)] = inputs
                action_service.store.set(
                    uid=user_code_id,
                    credentials=context.node.signing_key.verify_key,
                    syft_object=dict_object,
                )
            else:
                return res

        return Ok(Ok(True))

    @service_method(path="oblv.enclave_transfer", name="enclave_transfer")
    def enclave_transfer(
        self,
        context: AuthedServiceContext,
        data: UID,
        deployment_id: str,
        oblv_client: OblvClient,
    ):
        request = EnclaveTransfer(
            deployment_id=deployment_id,
            data_id=data if isinstance(data, UID) else data.id,
            user_verify_key=context.credentials,
            created_at=datetime.utcnow().strftime("%d/%m/%Y %H:%M:%S"),
            oblv_client=oblv_client,
        )
        res = self.enclave_transfer_stash.set(request)
        if res.is_err():
            return SyftError(message=str(res.err()))

        # relative
        from ..linked_obj import LinkedObject
        from ..request import EnumMutation
        from ..request import SubmitRequest
        from ..request_service import RequestService

        linked_obj = LinkedObject.from_obj(
            request, node_uid=context.node.id, stash_name="enclave_transfer_stash"
        )
        REQUEST_APPROVE = EnumMutation.from_obj(
            linked_obj=linked_obj,
            attr_name="status",
            approved_value=EnclaveTransferStatus.APPROVED,
            denied_value=EnclaveTransferStatus.DENIED,
            old_value=getattr(request, "status"),
        )
        request = SubmitRequest(changes=[REQUEST_APPROVE])
        method = context.node.get_service_method(RequestService.submit)
        result = method(context=context, request=request)

        # The Request service already returns either a SyftSuccess or SyftError
        return result

    @service_method(path="oblv.review_enclave_transfer", name="review_enclave_transfer")
    def review_enclave_transfer(
        self,
        context: AuthedServiceContext,
        request_id: UID,
        approve: bool,
        reason: str = "",
    ) -> Result[Ok, Err]:
        # TODO 🟣 Check for permission after it is fully integrated

        request = self.enclave_transfer_stash.get_by_uid(request_id)
        if request.is_ok():
            # 🟡 TODO: To Remove double nesting of result variable in Collection access
            request = request.ok().ok()
        else:
            return request.err()

        request.reason = reason
        request.reviewed_at = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        # request.reviewed_by =

        # Fetch private data from action store if the code task is approved
        if approve:
            # TODO 🟣 call action method to transfer data
            pass

        request.status = approve

        # Update task status back to DB
        res = self.enclave_transfer_stash.update(request)

        # If we are in the Enclave execution and have metadata for enclaves
        # Sent the task status to the connected enclave

        if res.is_ok():
            return Ok(
                SyftSuccess(
                    message=f"Request: {request_id}  - {'Approved' if approve else 'Denied'}"
                )
            )
        return res.err()

    @service_method(path="oblv.transfer.get_all", name="get_all")
    def get_all(
        self, context: AuthedServiceContext
    ) -> Union[List[UserCode], SyftError]:
        """Get a Dataset"""
        result = self.enclave_transfer_stash.get_all()
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())


# Checks if the given user code would  propogate value to enclave on acceptance
def check_enclave_transfer(
    user_code: UserCode, value: UserCodeStatus, context: ChangeContext
):
    if (
        isinstance(user_code.enclave_metadata, OblvMetadata)
        and value == UserCodeStatus.EXECUTE
    ):
        method = context.node.get_service_method(OblvService.get_api_for)

        api = method(
            user_code.enclave_metadata,
            context.node.signing_key,
        )
        # send data of the current node to enclave
        user_node_view = UserNodeView(
            node_name=context.node.name, verify_key=context.node.signing_key.verify_key
        )
        inputs = user_code.input_policy.inputs[user_node_view]
        action_service = context.node.get_service("actionservice")
        for var_name, uid in inputs.items():
            action_object = action_service.store.get(
                uid=uid, credentials=context.node.signing_key.verify_key
            )
            if action_object.is_err():
                return action_object
            inputs[var_name] = action_object.ok()

        res = api.services.oblv.send_user_code_inputs_to_enclave(
            user_code_id=user_code.id, inputs=inputs, node_name=context.node.name
        )

        return res
    else:
        return Ok()


TYPE_TO_SERVICE[EnclaveTransfer] = OblvService
SERVICE_TO_TYPES[OblvService].update({EnclaveTransfer})
