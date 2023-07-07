# stdlib
from base64 import encodebytes
import os
import random
import subprocess  # nosec
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import cast

# third party
from oblv_ctl import OblvClient
import requests
from result import Err
from result import Ok
from result import Result

# relative
from ...client.api import SyftAPI
from ...client.client import HTTPConnection
from ...client.client import Routes
from ...node.credentials import SyftSigningKey
from ...node.credentials import SyftVerifyKey
from ...serde.deserialize import _deserialize as deserialize
from ...serde.serializable import serializable
from ...service.action.action_object import ActionObject
from ...service.code.user_code import UserCodeStatus
from ...service.context import AuthedServiceContext
from ...service.response import SyftError
from ...service.service import AbstractService
from ...service.service import service_method
from ...service.user.user_roles import GUEST_ROLE_LEVEL
from ...store.document_store import DocumentStore
from ...types.uid import UID
from ...util.util import find_available_port
from .constants import DOMAIN_CONNECTION_PORT
from .constants import LOCAL_MODE
from .deployment_client import OblvMetadata
from .exceptions import OblvEnclaveError
from .exceptions import OblvProxyConnectPCRError
from .oblv_keys import OblvKeys
from .oblv_keys_stash import OblvKeysStash

# caches the connection to Enclave using the deployment ID
OBLV_PROCESS_CACHE: Dict[str, List] = {}


def connect_to_enclave(
    oblv_keys_stash: OblvKeysStash,
    verify_key: SyftVerifyKey,
    oblv_client: OblvClient,
    deployment_id: str,
    connection_port: int,
    oblv_key_name: str,
) -> Optional[subprocess.Popen]:
    global OBLV_PROCESS_CACHE
    if deployment_id in OBLV_PROCESS_CACHE:
        process = OBLV_PROCESS_CACHE[deployment_id][0]
        if process.poll() is None:
            return process
        # If the process has been terminated create a new connection
        del OBLV_PROCESS_CACHE[deployment_id]

    # Always create key file each time, which ensures consistency when there is key change in database
    create_keys_from_db(
        oblv_keys_stash=oblv_keys_stash,
        verify_key=verify_key,
        oblv_key_name=oblv_key_name,
    )
    oblv_key_path = os.path.expanduser(os.getenv("OBLV_KEY_PATH", "~/.oblv"))

    public_file_name = oblv_key_path + "/" + oblv_key_name + "_public.der"
    private_file_name = oblv_key_path + "/" + oblv_key_name + "_private.der"

    depl = oblv_client.deployment_info(deployment_id)
    if depl.is_deleted:
        raise OblvEnclaveError(
            "User cannot connect to this deployment, as it is no longer available."
        )
    if depl.is_dev_env:
        process = subprocess.Popen(  # nosec
            [
                "oblv",
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
                "oblv",
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
    return None


def make_request_to_enclave(
    oblv_keys_stash: OblvKeysStash,
    verify_key: SyftVerifyKey,
    deployment_id: str,
    oblv_client: OblvClient,
    request_method: Callable,
    connection_string: str,
    connection_port: int,
    oblv_key_name: str,
    params: Optional[Dict] = None,
    files: Optional[Dict] = None,
    data: Optional[Dict] = None,
    json: Optional[Dict] = None,
) -> Any:
    if not LOCAL_MODE:
        _ = connect_to_enclave(
            oblv_keys_stash=oblv_keys_stash,
            verify_key=verify_key,
            oblv_client=oblv_client,
            deployment_id=deployment_id,
            connection_port=connection_port,
            oblv_key_name=oblv_key_name,
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


def create_keys_from_db(
    oblv_keys_stash: OblvKeysStash, verify_key: SyftVerifyKey, oblv_key_name: str
):
    oblv_key_path = os.path.expanduser(os.getenv("OBLV_KEY_PATH", "~/.oblv"))

    os.makedirs(oblv_key_path, exist_ok=True)
    # Temporary new key name for the new service

    keys = oblv_keys_stash.get_all(verify_key)
    if keys.is_ok():
        keys = keys.ok()[0]
    else:
        return keys.err()

    f_private = open(oblv_key_path + "/" + oblv_key_name + "_private.der", "w+b")
    f_private.write(keys.private_key)
    f_private.close()
    f_public = open(oblv_key_path + "/" + oblv_key_name + "_public.der", "w+b")
    f_public.write(keys.public_key)
    f_public.close()


def generate_oblv_key(oblv_key_name: str) -> Tuple[bytes]:
    oblv_key_path = os.path.expanduser(os.getenv("OBLV_KEY_PATH", "~/.oblv"))
    os.makedirs(oblv_key_path, exist_ok=True)

    result = subprocess.run(  # nosec
        [
            "oblv",
            "keygen",
            "--key-name",
            oblv_key_name,
            "--output",
            oblv_key_path,
        ],
        capture_output=True,
    )

    if result.stderr:
        raise Err(
            subprocess.CalledProcessError(  # nosec
                returncode=result.returncode, cmd=result.args, stderr=result.stderr
            )
        )
    f_private = open(oblv_key_path + "/" + oblv_key_name + "_private.der", "rb")
    private_key = f_private.read()
    f_private.close()
    f_public = open(oblv_key_path + "/" + oblv_key_name + "_public.der", "rb")
    public_key = f_public.read()
    f_public.close()

    return (public_key, private_key)


@serializable()
class OblvService(AbstractService):
    store: DocumentStore
    oblv_keys_stash: OblvKeysStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.oblv_keys_stash = OblvKeysStash(store=store)

    @service_method(path="oblv.create_key", name="create_key", roles=GUEST_ROLE_LEVEL)
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

        res = self.oblv_keys_stash.set(context.credentials, oblv_keys)

        if res.is_ok():
            return Ok(
                "Successfully created a new public/private RSA key-pair on the domain node"
            )
        return res.err()

    @service_method(
        path="oblv.get_public_key", name="get_public_key", roles=GUEST_ROLE_LEVEL
    )
    def get_public_key(
        self,
        context: AuthedServiceContext,
    ) -> Result[Ok, Err]:
        "Retrieves the public key present on the Domain Node."

        if len(self.oblv_keys_stash):
            # retrieve the public key from the stash using the node's verify key
            # as the public should be accessible to all the users
            oblv_keys = self.oblv_keys_stash.get_all(context.node.verify_key)
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
        worker_name: str,
    ) -> SyftAPI:
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

            # To identify if we are in docker container
            if "CONTAINER_HOST" in os.environ:
                connection_string = connection_string.replace(
                    "127.0.0.1", "host.docker.internal"
                )

        params = {"verify_key": str(signing_key.verify_key)}
        req = make_request_to_enclave(
            connection_string=connection_string + Routes.ROUTE_API.value,
            deployment_id=deployment_id,
            oblv_client=oblv_client,
            oblv_keys_stash=self.oblv_keys_stash,
            verify_key=signing_key.verify_key,
            request_method=requests.get,
            connection_port=port,
            oblv_key_name=worker_name,
            params=params,
        )

        obj = deserialize(req.content, from_bytes=True)
        # TODO 🟣 Retrieve of signing key of user after permission  is fully integrated
        obj.signing_key = signing_key
        obj.connection = HTTPConnection(url=connection_string)
        return cast(SyftAPI, obj)

    @service_method(
        path="oblv.send_user_code_inputs_to_enclave",
        name="send_user_code_inputs_to_enclave",
        roles=GUEST_ROLE_LEVEL,
    )
    def send_user_code_inputs_to_enclave(
        self,
        context: AuthedServiceContext,
        user_code_id: UID,
        inputs: Dict,
        node_name: str,
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
            verify_key=context.credentials,
        )
        if res.is_err():
            return res
        user_code.status = res.ok()
        user_code_service.update_code_state(context=context, code_item=user_code)

        if not action_service.exists(context=context, obj_id=user_code_id):
            dict_object = ActionObject.from_obj({})
            dict_object.id = user_code_id
            dict_object[str(context.credentials)] = inputs
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
                dict_object[str(context.credentials)] = inputs
                action_service.store.set(
                    uid=user_code_id,
                    credentials=context.node.verify_key,
                    syft_object=dict_object,
                )
            else:
                return res

        return Ok(Ok(True))
