# future
from __future__ import annotations

# stdlib
from datetime import datetime
import os
from signal import SIGTERM
import subprocess  # nosec
import sys
import time
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import TYPE_CHECKING

# third party
from oblv_ctl import OblvClient
from pydantic import validator
import requests

# relative
from ...client.api import SyftAPI
from ...client.client import SyftClient
from ...client.client import login
from ...client.client import login_as_guest
from ...client.enclave_client import EnclaveMetadata
from ...serde.serializable import serializable
from ...types.uid import UID
from ...util.util import bcolors
from .constants import LOCAL_MODE
from .exceptions import OblvEnclaveError
from .exceptions import OblvUnAuthorizedError
from .oblv_proxy import check_oblv_proxy_installation_status

if TYPE_CHECKING:
    # relative
    from ...service.code.user_code import SubmitUserCode


@serializable()
class OblvMetadata(EnclaveMetadata):
    """Contains Metadata to connect to Oblivious Enclave"""

    deployment_id: Optional[str]
    oblv_client: Optional[OblvClient]

    @validator("deployment_id")
    def check_valid_deployment_id(cls, deployment_id: str) -> str:
        if not deployment_id and not LOCAL_MODE:
            raise ValueError(
                f"Deployment ID should be a valid string: {deployment_id}"
                + "in cloud deployment of enclave"
                + "For testing set the LOCAL_MODE variable in constants.py"
            )
        return deployment_id

    @validator("oblv_client")
    def check_valid_oblv_client(cls, oblv_client: OblvClient) -> OblvClient:
        if not oblv_client and not LOCAL_MODE:
            raise ValueError(
                f"Oblivious Client should be a valid client: {oblv_client}"
                + "in cloud deployment of enclave"
                + "For testing set the LOCAL_MODE variable in constants.py"
            )
        return oblv_client


class DeploymentClient:
    deployment_id: str
    key_name: str
    domain_clients: List[SyftClient]  # List of domain client objects
    oblv_client: OblvClient = None
    __conn_string: str
    __logs: Any
    __process: Any
    __enclave_client: SyftClient

    def __init__(
        self,
        domain_clients: List[SyftClient],
        deployment_id: str,
        oblv_client: Optional[OblvClient] = None,
        key_name: Optional[str] = None,
        api: Optional[SyftAPI] = None,
    ):
        if not domain_clients:
            raise Exception(
                "domain_clients should be populated with valid domain nodes"
            )
        self.deployment_id = deployment_id
        self.key_name = key_name
        self.oblv_client = oblv_client
        self.domain_clients = domain_clients
        self.__conn_string = ""
        self.__process = None
        self.__logs = None
        self._api = api
        self.__enclave_client = None

    def make_request_to_enclave(
        self,
        request_method: Callable,
        connection_string: str,
        params: Optional[Dict] = None,
        files: Optional[Dict] = None,
        data: Optional[Dict] = None,
        json: Optional[Dict] = None,
    ) -> Any:
        header = {}
        if LOCAL_MODE:
            header["x-oblv-user-name"] = "enclave_test"
            header["x-oblv-user-role"] = "user"
        else:
            depl = self.oblv_client.deployment_info(self.deployment_id)
            if depl.is_deleted:
                raise Exception(
                    "User cannot connect to this deployment, as it is no longer available."
                )
        return request_method(
            connection_string,
            headers=header,
            params=params,
            files=files,
            data=data,
            json=json,
        )

    def set_conn_string(self, url: str) -> None:
        self.__conn_string = url

    def initiate_connection(self, connection_port: int = 3030) -> None:
        if LOCAL_MODE:
            self.__conn_string = f"http://127.0.0.1:{connection_port}"
            return
        check_oblv_proxy_installation_status()
        self.close_connection()  # To close any existing connections
        public_file_name = os.path.join(
            os.path.expanduser("~"),
            ".ssh",
            self.key_name,
            self.key_name + "_public.der",
        )
        private_file_name = os.path.join(
            os.path.expanduser("~"),
            ".ssh",
            self.key_name,
            self.key_name + "_private.der",
        )
        log_file_name = os.path.join(
            os.path.expanduser("~"),
            ".oblv_syft_logs",
            "proxy_logs_" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ".log",
        )
        # Creating directory if not exist
        os.makedirs(os.path.dirname(log_file_name), exist_ok=True)
        log_file = open(log_file_name, "wb")
        depl = self.oblv_client.deployment_info(self.deployment_id)
        if depl.is_deleted:
            raise Exception(
                "User cannot connect to this deployment, as it is no longer available."
            )
        try:
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
                    stdout=log_file,
                    stderr=log_file,
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
                    stdout=log_file,
                    stderr=log_file,
                )
            with open(log_file_name) as log_file_read:
                while True:
                    log_line = log_file_read.readline()
                    if "Error:  Invalid PCR Values" in log_line:
                        raise Exception("PCR Validation Failed")
                    if "Only one usage of each socket address" in log_line:
                        raise Exception(
                            "Another oblv proxy instance running. Either close that connection"
                            + "or change the *connection_port*"
                        )
                    elif "error" in log_line.lower():
                        raise Exception(log_line)
                    elif "listening on" in log_line:
                        break
        except Exception as e:
            raise e
        else:
            print(
                f"Successfully connected to proxy on port {connection_port}. The logs can be found at {log_file_name}"
            )
        self.__conn_string = f"http://127.0.0.1:{connection_port}"
        self.__logs = log_file_name
        self.__process = process
        return

    def register(
        self,
        name: str,
        email: str,
        password: str,
        institution: Optional[str] = None,
        website: Optional[str] = None,
    ):
        self.check_connection_string()
        guest_client = login_as_guest(url=self.__conn_string)
        return guest_client.register(
            name=name,
            email=email,
            password=password,
            institution=institution,
            website=website,
        )

    def login(
        self,
        email: str,
        password: str,
    ) -> None:
        self.check_connection_string()
        self.__enclave_client = login(
            url=self.__conn_string, email=email, password=password
        )

    def check_connection_string(self) -> None:
        if not self.__conn_string:
            raise Exception(
                "Either proxy not running or not initiated using syft."
                + " Run the method initiate_connection to initiate the proxy connection"
            )

    def sanity_check_oblv_response(self, req: requests.Response) -> str:
        if req.status_code == 401:
            raise OblvUnAuthorizedError()
        elif req.status_code == 400:
            raise OblvEnclaveError(req.json()["detail"])
        elif req.status_code == 422:
            print(req.text)
            # ToDo - Update here
        elif req.status_code != 200:
            raise OblvEnclaveError(
                f"Failed to perform the operation  with status {req.status_code}, {req.content!r}"
            )
        return "Failed"

    def request_code_execution(self, code: SubmitUserCode) -> Any:
        # relative
        from ...service.code.user_code import SubmitUserCode

        if not isinstance(code, SubmitUserCode):
            raise Exception(
                f"The input code should be of type: {SubmitUserCode} got:{type(code)}"
            )

        enclave_metadata = OblvMetadata(
            deployment_id=self.deployment_id, oblv_client=self.oblv_client
        )

        code_id = UID()
        code.id = code_id
        code.enclave_metadata = enclave_metadata

        for domain_client in self.domain_clients:
            domain_client.code.request_code_execution(code=code)
            print(f"Sent code execution request to {domain_client.name}")

        res = self.api.services.code.request_code_execution(code=code)
        print(f"Execution will be done on {self.__enclave_client.name}")

        return res

    @property
    def api(self) -> SyftAPI:
        if not self.__enclave_client:
            raise Exception("Kindly login or register with the enclave")

        return self.__enclave_client.api

    def close_connection(self) -> Optional[str]:
        if self.check_proxy_running():
            os.kill(self.__process.pid, SIGTERM)
            return None
        else:
            return "No Proxy Connection Running"

    def check_proxy_running(self) -> bool:
        if self.__process is not None:
            if self.__process.poll() is not None:
                return False
            else:
                return True
        return False

    def fetch_current_proxy_logs(
        self, follow: bool = False, tail: bool = False
    ) -> None:
        """Returns the logs of the running enclave instance

        Args:
            follow (bool, optional): To follow the logs as they grow. Defaults to False.
            tail (bool, optional): Only show the new generated logs.
                To be used only when follow is True. Defaults to False.
        """
        if self.__logs is None:
            print(
                bcolors.RED
                + bcolors.BOLD
                + "Exception"
                + bcolors.BLACK
                + bcolors.ENDC
                + ": Logs not initiated",
                file=sys.stderr,
            )
        log_file = open(self.__logs)
        if not follow:
            print(log_file.read())
        else:
            if tail:
                log_file.seek(0, 2)
            while True:
                line = log_file.readline()
                if not line:
                    time.sleep(0.1)
                    continue
                print(line)


# Todo - Method to check if proxy is running
# Todo
