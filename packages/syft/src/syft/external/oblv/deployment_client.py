# future
from __future__ import annotations

# stdlib
from datetime import datetime
import json
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
from typing import cast

# third party
from oblv import OblvClient
from pydantic import BaseModel
from pydantic import validator
import requests

# relative
from ...core.node.new.api import SyftAPI
from ...core.node.new.client import HTTPConnection
from ...core.node.new.client import Routes
from ...core.node.new.client import SyftSigningKey
from ...core.node.new.deserialize import _deserialize as deserialize
from ...core.node.new.node_metadata import EnclaveMetadata
from ...core.node.new.serializable import serializable
from ...core.node.new.uid import UID
from ...util import bcolors
from .constants import LOCAL_MODE
from .exceptions import OblvEnclaveError
from .exceptions import OblvUnAuthorizedError
from .oblv_proxy import check_oblv_proxy_installation_status

if TYPE_CHECKING:
    # relative
    from ...core.node.new.user_code import SubmitUserCode


@serializable(recursive_serde=True)
class OblvMetadata(EnclaveMetadata, BaseModel):
    """Contains Metadata to connect to Oblivious Enclave"""

    class Config:
        arbitrary_types_allowed = True

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
    user_key_name: str
    domain_clients: List[Any] = []  # List of domain client objects
    oblv_client: OblvClient = None
    __conn_string: str
    __logs: Any
    __process: Any

    def __init__(
        self,
        domain_clients: List[Any],
        user_key_name: str,
        deployment_id: str,
        oblv_client: Optional[OblvClient] = None,
        api: Optional[SyftAPI] = None,
    ):
        if not domain_clients:
            raise Exception(
                "domain_clients should be populated with valid domain nodes"
            )
        self.deployment_id = deployment_id
        self.user_key_name = user_key_name
        self.oblv_client = oblv_client
        self.domain_clients = domain_clients
        self.__conn_string = ""
        self.__process = None
        self.__logs = None
        self._api = api

    def make_request_to_enclave(
        self,
        request_method: Callable,
        connection_string: str,
        params: Optional[Dict] = None,
        files: Optional[Dict] = None,
        data: Optional[Dict] = None,
        json: Optional[Dict] = None,
    ) -> Any:
        print(data)
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
            self.user_key_name,
            self.user_key_name + "_public.der",
        )
        private_file_name = os.path.join(
            os.path.expanduser("~"),
            ".ssh",
            self.user_key_name,
            self.user_key_name + "_private.der",
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
            log_file_read = open(log_file_name, "r")
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
        from ...core.node.new.user_code import SubmitUserCode

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
            domain_client.api.services.code.request_code_execution(code=code)

        res = self.api.services.code.request_code_execution(code=code)

        return res

    def get_uploaded_datasets(self) -> Dict:
        self.check_connection_string()

        req = self.make_request_to_enclave(
            requests.get, connection_string=self.__conn_string + "/tensor/dataset/list"
        )
        self.sanity_check_oblv_response(req)
        return req.json()  # This is the publish_request_id

    def publish_action(
        self, action: str, arguments: List, *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        self.check_connection_string()
        file = None
        if len(arguments) == 2 and arguments[1]["type"] == "tensor":
            file = arguments[1]["value"]
            arguments[1]["value"] = "file"
        body = {
            "inputs": json.dumps(arguments),
            "args": json.dumps(args),
            "kwargs": json.dumps(kwargs),
        }

        req = self.make_request_to_enclave(
            requests.post,
            connection_string=self.__conn_string + f"/tensor/action?op={action}",
            data=body,
            files={"file": file},
        )

        self.sanity_check_oblv_response(req)

        # Status code 200
        # TODO - Remove this after oblv proxy is resolved
        data = req.json()
        if isinstance(data, dict) and data.get("detail") is not None:
            raise OblvEnclaveError(data["detail"])
        return data

    def request_publish(self, dataset_id: UID, sigma: float = 0.5) -> Dict[str, Any]:
        self.check_connection_string()

        req = self.make_request_to_enclave(
            requests.post,
            connection_string=self.__conn_string + "/tensor/publish/request",
            json={"dataset_id": dataset_id, "sigma": sigma},
        )
        self.sanity_check_oblv_response(req)

        # Status code 200
        # TODO - Remove this after oblv proxy is resolved

        data = req.json()
        if type(data) == dict and data.get("detail") is not None:
            raise OblvEnclaveError(data["detail"])
        # Here data is publish_request_id
        for domain_client in self.domain_clients:
            domain_client.oblv.publish_budget(
                deployment_id=self.deployment_id,
                publish_request_id=data,
                client=self.oblv_client,
            )
        return data

    def _get_api(self) -> SyftAPI:
        self.check_connection_string()
        signing_key = SyftSigningKey.generate()

        params = {"verify_key": str(signing_key.verify_key)}
        req = self.make_request_to_enclave(
            requests.get,
            connection_string=self.__conn_string + Routes.ROUTE_API.value,
            params=params,
        )
        self.sanity_check_oblv_response(req)
        obj = deserialize(req.content, from_bytes=True)
        # TODO 🟣 Retrieve of signing key of user after permission  is fully integrated
        obj.signing_key = signing_key
        obj.connection = HTTPConnection(self.__conn_string)
        return cast(SyftAPI, obj)

    # public attributes

    def _set_api(self) -> None:
        _api = self._get_api()
        # APIRegistry.set_api_for(node_uid=self.id, api=_api)
        self._api = _api

    @property
    def api(self) -> SyftAPI:
        if self._api is None:
            self._set_api()

        return cast(SyftAPI, self._api)

    def refresh(self) -> None:
        self._set_api()

    def check_publish_request_status(self, publish_request_id: UID) -> None:
        self.check_connection_string()

        req = self.make_request_to_enclave(
            requests.get,
            connection_string=self.__conn_string
            + "/tensor/publish/result_ready?publish_request_id="
            + str(publish_request_id),
        )

        self.sanity_check_oblv_response(req)

        result = req.json()
        if isinstance(result, str):
            print("Not yet Ready")
        elif not isinstance(result, dict):
            for domain_client in self.domain_clients:
                domain_client.oblv.publish_request_budget_deduction(
                    deployment_id=self.deployment_id,
                    publish_request_id=publish_request_id,
                    client=self.oblv_client,
                    budget_to_deduct=result,
                )
            print("Result is ready")  # This is the publish_request_id
        else:
            # TODO - Remove this after oblv proxy is resolved
            if result.get("detail") is not None:
                raise OblvEnclaveError(result["detail"])

    def fetch_result(self, publish_request_id: UID) -> Dict[str, Any]:
        self.check_connection_string()

        req = self.make_request_to_enclave(
            requests.get,
            connection_string=self.__conn_string
            + "/tensor/publish/result?request_id="
            + str(publish_request_id),
        )
        self.sanity_check_oblv_response(req)
        return req.json()

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
        log_file = open(self.__logs, "r")
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
