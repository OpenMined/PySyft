# stdlib
from base64 import encodebytes
import os
import subprocess
from typing import Callable
from typing import Dict
from typing import Optional

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

DOMAIN_CONNECTION_PORT = str(os.getenv("DOMAIN_CONNECTION_PORT", 3030))


def connect_to_enclave(
    oblv_keys_stash: OblvKeysStash, oblv_client: OblvClient, deployment_id: str
) -> subprocess.Popen:

    # Always create key file each time, which ensures consistency when there is key change in database
    create_keys_from_db(oblv_keys_stash)
    key_path = os.getenv("OBLV_KEY_PATH", "/app/content")
    # TODO 🟣 Setting a custom name for the key as the 0.7 oblv_service uses the same code
    # Should be modified to use environment variables when old service code is removed.
    key_name = "new_oblv_key"
    cli = oblv_client
    public_file_name = key_path + "/" + key_name + "_public.der"
    private_file_name = key_path + "/" + key_name + "_private.der"
    depl = cli.deployment_info(deployment_id)
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
                DOMAIN_CONNECTION_PORT,
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
                DOMAIN_CONNECTION_PORT,
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

    return process


def make_request_to_enclave(
    oblv_keys_stash: OblvKeysStash,
    deployment_id: str,
    oblv_client: OblvClient,
    request_method: Callable,
    connection_string: str,
    params: Optional[Dict] = None,
    files: Optional[Dict] = None,
    data: Optional[Dict] = None,
    json: Optional[Dict] = None,
):
    if not LOCAL_MODE:
        process = connect_to_enclave(
            oblv_keys_stash=oblv_keys_stash,
            oblv_client=oblv_client,
            deployment_id=deployment_id,
        )
        req = request_method(
            connection_string,
            params=params,
            files=files,
            data=data,
            json=json,
        )
        process.kill()
        process.wait(1)
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

    # TODO 🟣 Setting a custom name for the key as the 0.7 oblv_service uses the same code
    # Should be modified when old service code is removed.
    file_name = "new_oblv_key"

    keys = oblv_keys_stash.get_by_id(id_int=1)
    if keys.is_ok():
        keys = keys.ok().ok()

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


@serializable(recursive_serde=True)
class OblvService(AbstractService):
    document_store: DocumentStore
    oblv_keys_stash: OblvKeysStash

    def __init__(self, document_store: DocumentStore) -> None:
        self.document_store = document_store
        self.oblv_keys_stash = OblvKeysStash(store=document_store)

    @service_method(path="oblv.create_key", name="create_key")
    def create_key(
        self,
        context: AuthedServiceContext,
    ) -> Result[Ok, Err]:
        """Domain Public/Private Key pair creation"""
        # TODO 🟣 Check for permission after it is fully integrated
        file_path = os.getenv("OBLV_KEY_PATH", "/app/content")
        # file_name = os.getenv("OBLV_KEY_NAME", "oblv_key")
        # TODO 🟣 Setting a custom name for the key as the 0.7 oblv_service uses the same code
        # Should be modified when old service code is removed.
        file_name = "new_oblv_key"
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

        self.oblv_keys_stash.delete_by_id(1)
        oblv_keys = OblvKeys(id_int=1, public_key=public_key, private_key=private_key)

        res = self.oblv_keys_stash.set(oblv_keys)

        if res.is_ok():
            return Ok(
                "Successfully created a new public/private key pair on the domain node"
            )
        return res.err()

    @service_method(path="oblv.get_key", name="get_key")
    def get_public_key(
        self,
        context: AuthedServiceContext,
    ) -> Result[Ok, Err]:

        row_exists = self.oblv_keys_stash.get_by_id(id_int=1).ok()

        if row_exists.is_ok() and not row_exists.ok():
            res = self.create_key(context)
            if res.is_err():
                return res.err()

        oblv_keys = self.oblv_keys_stash.get_by_id(id_int=1)
        if oblv_keys.is_ok():
            oblv_keys = oblv_keys.ok().ok()
            public_key_str = (
                encodebytes(oblv_keys.public_key).decode("UTF-8").replace("\n", "")
            )
            return Ok(public_key_str)

        return Err(
            "Oblv Keys not present for the domain node, Kindly request the admin to create a new one"
        )
