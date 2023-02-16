# stdlib
from base64 import encodebytes
import os
import subprocess
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

# third party
from oblv import OblvClient
from result import Err
from result import Ok
from result import Result

# relative
from .....oblv.constants import LOCAL_MODE
from ....common.serde.serializable import serializable
from ...common.exceptions import OblvEnclaveError
from ...common.exceptions import OblvProxyConnectPCRError
from ..context import AuthedServiceContext
from ..document_store import DocumentStore
from ..service import AbstractService
from ..service import service_method
from .oblv_keys import OblvKeys
from .oblv_keys_stash import OblvKeysStash

# caches the connection to Enclave using the deployment ID
OBLV_PROCESS_CACHE: Dict[str, List] = {}


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
            connection_string.replace("127.0.0.1", "host.docker.internal"),
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

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.oblv_keys_stash = OblvKeysStash(store=store)

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

    # TODO 🟣 Temporary method to help in testing an enclave
    # as with the in-memory we do not have persistent database
    # due to which oblv keys are generated during each hot reload
    # These test method should be removed when we have persistent database in the worker
    @service_method(path="oblv.retrieve_key", name="retrieve_key")
    def retrieve_key(
        self,
        context: AuthedServiceContext,
    ) -> Result[Ok, Err]:
        res = self.oblv_keys_stash.get_all()[0]
        assert isinstance(res, OblvKeys)
        return res

    @service_method(path="oblv.override_key", name="override_key")
    def override_key(
        self,
        context: AuthedServiceContext,
        oblv_key: OblvKeys,
    ) -> Result[Ok, Err]:
        self.oblv_keys_stash.clear()
        res = self.oblv_keys_stash.set(oblv_key)

        if res.is_ok():
            return Ok("Successfully overrided key")
        return res.err()
