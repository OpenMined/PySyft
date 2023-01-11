# stdlib
from datetime import datetime
import json
import os
from signal import SIGTERM
import subprocess  # nosec
import sys
import time
from typing import Any
from typing import List

# third party
from oblv import OblvClient
import requests

# relative
from ..core.node.common.exceptions import OblvEnclaveError
from ..core.node.common.exceptions import OblvUnAuthorizedError
from ..util import bcolors
from .oblv_proxy import check_oblv_proxy_installation_status


class DeploymentClient:

    # __attr_allowlist__ = ["deployment_id", "user_key_name", "client", "oblv_client"]

    deployment_id: str
    user_key_name: str
    client: List[Any] = []  # List of domain client objects
    oblv_client: OblvClient = None
    __conn_string: str
    __logs: Any
    __process: Any

    def __init__(
        self,
        domain_clients: List[Any],
        deployment_id: str = None,
        oblv_client: OblvClient = None,
        user_key_name: str = "",
    ):
        self.deployment_id = deployment_id
        self.user_key_name = user_key_name
        self.oblv_client = oblv_client
        self.client = domain_clients
        self.__conn_string = ""
        self.__process = None
        self.__logs = None

    def set_conn_string(self, URL):
        self.__conn_string = URL

    def initiate_connection(self, connection_port: int = 3032):
        if check_oblv_proxy_installation_status() is None:
            return
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
                d = log_file_read.readline()
                if d.__contains__("Error:  Invalid PCR Values"):
                    raise Exception("PCR Validation Failed")
                if d.__contains__("Only one usage of each socket address"):
                    raise Exception(
                        "Another oblv proxy instance running. Either close that connection"
                        + "or change the *connection_port*"
                    )
                elif d.lower().__contains__("error"):
                    raise Exception(d)
                elif d.__contains__("listening on"):
                    break
        except Exception as e:
            print("Could not connect to Proxy")
            raise e
        else:
            print(
                "Successfully connected to proxy on port {}. The logs can be found at {}".format(
                    connection_port, log_file_name
                )
            )
        self.__conn_string = "http://127.0.0.1:" + str(connection_port)
        self.__logs = log_file_name
        self.__process = process
        return

    def get_uploaded_datasets(self):
        if len(self.client) == 0:
            raise Exception(
                "No Domain Clients added. Set the propert *client* with the list of your domain logins"
            )
        if self.__conn_string is None:
            raise Exception(
                "proxy not running. Use the method connect_oblv_proxy to start the proxy."
            )
        elif self.__conn_string == "":
            raise Exception(
                "Either proxy not running or not initiated using syft. Run the method initiate_connection"
                + "to initiate the proxy connection"
            )
        depl = self.oblv_client.deployment_info(self.deployment_id)
        if depl.is_deleted:
            raise Exception(
                "User cannot connect to this deployment, as it is no longer available."
            )
        req = requests.get(self.__conn_string + "/tensor/dataset/list")
        if req.status_code == 401:
            raise OblvUnAuthorizedError()
        elif req.status_code == 400:
            raise OblvEnclaveError(req.json()["detail"])
        elif req.status_code == 422:
            print(req.text)
            # ToDo - Update here
            return "Failed"
        elif req.status_code != 200:
            raise OblvEnclaveError(
                "Request to publish dataset failed with status {}".format(
                    req.status_code
                )
            )
        return req.json()  # This is the publish_request_id

    def publish_action(self, action: str, arguments, *args, **kwargs):
        if self.__conn_string is None:
            raise Exception(
                "proxy not running. Use the method connect_oblv_proxy to start the proxy."
            )
        elif self.__conn_string == "":
            raise Exception(
                "Either proxy not running or not initiated using syft. Run the method initiate_connection"
                + "to initiate the proxy connection"
            )
        depl = self.oblv_client.deployment_info(self.deployment_id)
        if depl.is_deleted:
            raise Exception(
                "User cannot connect to this deployment, as it is no longer available."
            )
        file = None
        if len(arguments) == 2 and arguments[1]["type"] == "tensor":
            file = arguments[1]["value"]
            arguments[1]["value"] = "file"
        body = {
            "inputs": json.dumps(arguments),
            "args": json.dumps(args),
            "kwargs": json.dumps(kwargs),
        }
        req = requests.post(
            self.__conn_string + "/tensor/action?op={}".format(action),
            data=body,
            files={"file": file},
        )
        # req = requests.post(self.__conn_string + "/tensor/action?op={}".format(action)
        # , data=body, files={"file": file},headers={"x-oblv-user-name": "vinal", "x-oblv-user-role": "user"})
        if req.status_code == 401:
            raise OblvUnAuthorizedError()
        elif req.status_code == 400:
            raise OblvEnclaveError(req.json()["detail"])
        elif req.status_code == 422:
            print(req.text)
            # ToDo - Update here
            return "Failed"
        elif req.status_code != 200:
            raise OblvEnclaveError(
                "Request to publish dataset failed with status {}".format(
                    req.status_code
                )
            )
        else:
            # Status code 200
            # TODO - Remove this after oblv proxy is resolved
            data = req.json()
            if type(data) == dict and data.get("detail") is not None:
                raise OblvEnclaveError(data["detail"])
            return data
        # #return req.json()

    def request_publish(self, dataset_id, sigma=0.5):
        if len(self.client) == 0:
            raise Exception(
                "No Domain Clients added. Set the propert *client* with the list of your domain logins"
            )
        if self.__conn_string is None:
            raise Exception(
                "proxy not running. Use the method connect_oblv_proxy to start the proxy."
            )
        elif self.__conn_string == "":
            raise Exception(
                "Either proxy not running or not initiated using syft. Run the method initiate_connection"
                + "to initiate the proxy connection"
            )
        depl = self.oblv_client.deployment_info(self.deployment_id)
        if depl.is_deleted:
            raise Exception(
                "User cannot connect to this deployment, as it is no longer available."
            )
        req = requests.post(
            self.__conn_string + "/tensor/publish/request",
            json={"dataset_id": dataset_id, "sigma": sigma},
        )
        if req.status_code == 401:
            raise OblvUnAuthorizedError()
        elif req.status_code == 400:
            raise OblvEnclaveError(req.json()["detail"])
        elif req.status_code == 422:
            print(req.text)
            # ToDo - Update here
            return "Failed"
        elif req.status_code != 200:
            raise OblvEnclaveError(
                "Request to publish dataset failed with status {}".format(
                    req.status_code
                )
            )
        else:
            # Status code 200
            # TODO - Remove this after oblv proxy is resolved

            data = req.json()
            if type(data) == dict and data.get("detail") is not None:
                raise OblvEnclaveError(data["detail"])
            # Here data is publish_request_id
            for o in self.client:
                o.oblv.publish_budget(
                    deployment_id=self.deployment_id,
                    publish_request_id=data,
                    client=self.oblv_client,
                )
            return data

    def check_publish_request_status(self, publish_request_id):
        if self.__conn_string is None:
            raise Exception(
                "proxy not running. Use the method connect_oblv_proxy to start the proxy."
            )
        elif self.__conn_string == "":
            raise Exception(
                "Either proxy not running or not initiated using syft. Run the method initiate_connection"
                + "to initiate the proxy connection"
            )
        depl = self.oblv_client.deployment_info(self.deployment_id)
        if depl.is_deleted:
            raise Exception(
                "User cannot connect to this deployment, as it is no longer available."
            )
        req = requests.get(
            self.__conn_string
            + "/tensor/publish/result_ready?publish_request_id="
            + publish_request_id
        )

        if req.status_code == 401:
            raise OblvUnAuthorizedError()
        elif req.status_code == 400:
            raise OblvEnclaveError(req.json()["detail"])
        elif req.status_code == 422:
            print(req.text)
            # ToDo - Update here
            return "Failed"
        elif req.status_code != 200:
            raise OblvEnclaveError(
                "Request to publish dataset failed with status {}".format(
                    req.status_code
                )
            )
        else:
            result = req.json()
            if type(result) == str:
                print("Not yet Ready")
            elif type(result) != dict:
                for o in self.client:
                    o.oblv.publish_request_budget_deduction(
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

    def fetch_result(self, publish_request_id):
        if self.__conn_string is None:
            raise Exception(
                "proxy not running. Use the method connect_oblv_proxy to start the proxy."
            )
        elif self.__conn_string == "":
            raise Exception(
                "Either proxy not running or not initiated using syft. Run the method initiate_connection"
                + "to initiate the proxy connection"
            )
        depl = self.oblv_client.deployment_info(self.deployment_id)
        if depl.is_deleted:
            raise Exception(
                "User cannot connect to this deployment, as it is no longer available."
            )
        req = requests.get(
            self.__conn_string
            + "/tensor/publish/result?request_id="
            + publish_request_id
        )
        if req.status_code == 401:
            raise OblvUnAuthorizedError()
        elif req.status_code == 400:
            raise OblvEnclaveError(req.json()["detail"])
        elif req.status_code == 422:
            print(req.text)
            # ToDo - Update here
            return "Failed"
        elif req.status_code != 200:
            raise OblvEnclaveError(
                "Request to publish dataset failed with status {}".format(
                    req.status_code
                )
            )
        return req.json()

    def close_connection(self):
        if self.check_proxy_running():
            os.kill(self.__process.pid, SIGTERM)
        else:
            return "No Proxy Connection Running"

    def check_proxy_running(self):
        if self.__process is not None:
            if self.__process.poll() is not None:
                return False
            else:
                return True
        return False

    def fetch_current_proxy_logs(self, follow=False, tail=False):
        """_summary_

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
