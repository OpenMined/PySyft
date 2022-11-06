# stdlib
import os
from signal import SIGILL
from signal import SIGTERM
import subprocess
from typing import Any
from typing import List

# third party
from oblv import OblvClient
import requests

# relative
from ..core.node.common.exceptions import OblvEnclaveError
from ..core.node.common.exceptions import OblvUnAuthorizedError
from .oblv_proxy import check_oblv_proxy_installation_status


class DeploymentClient():
    deployment_id: str
    user_key_name: str
    connection_port: int = 3032
    client: List[Any] = [] #List of domain client objects
    oblv_client: OblvClient = None
    conn_string: str
    pid: int
    
    def __init__(self,deployment_id: Any=None, oblv_client: OblvClient = None,user_key_name="" , connection_port=3032):
        self.deployment_id=deployment_id
        self.user_key_name = user_key_name
        self.connection_port = connection_port
        self.oblv_client = oblv_client
        self.conn_string = ""

    def initiate_connection(self):
        check_oblv_proxy_installation_status()
        public_file_name = os.path.join(os.path.expanduser('~'),'.ssh',self.user_key_name,self.user_key_name+'_public.der')
        private_file_name = os.path.join(os.path.expanduser('~'),'.ssh',self.user_key_name,self.user_key_name+'_private.der')
        depl = self.oblv_client.deployment_info(self.deployment_id)
        if depl.is_deleted==True:
            raise Exception("User cannot connect to this deployment, as it is no longer available.")
        process = subprocess.Popen([
            "oblv", "connect",
            "--private-key", private_file_name,
            "--public-key", public_file_name,
            "--url", depl.instance.service_url,
            "--port","443",
            "--lport",str(self.connection_port),
            "--disable-pcr-check"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.check_call
        while process.poll() is None:
            d = process.stderr.readline().decode()
            print(d)
            if d.__contains__("Error:  Invalid PCR Values"):
                raise Exception("PCR Validation Failed")
            elif d.__contains__("Error"):
                raise Exception(message=d)
            elif d.__contains__("listening on"):
                break
        self.conn_string = "http://127.0.0.1:"+str(self.connection_port)
        self.pid = process.pid
        return

    def publish_action(self,action: str, arguments):
        if self.conn_string==None:
            raise Exception("proxy not running. Use the method connect_oblv_proxy to start the proxy.")
        elif self.conn_string=="":
            raise Exception("Either proxy not running or not initiated using syft. Set the conn_string to get started or run the method initiate_connection to initiate the proxy connection")
        depl = self.oblv_client.deployment_info(self.deployment_id)
        if depl.is_deleted==True:
            raise Exception("User cannot connect to this deployment, as it is no longer available.")
        req = requests.post(self.conn_string + "/tensor/action?op={}".format(action)
                            , headers={"x-oblv-user-name":"vinal", "x-oblv-user-role": "user"}
                            , json=arguments)
        if req.status_code==401:
            raise OblvUnAuthorizedError()
        elif req.status_code == 400:
            raise OblvEnclaveError(req.json()["detail"])
        elif req.status_code==422:
            print(req.text)
            #ToDo - Update here
            return "Failed"
        elif req.status_code!=200:
            raise OblvEnclaveError("Request to publish dataset failed with status {}".format(req.status_code))        
        return req.json()

    def request_publish(self, dataset_id, sigma = 0.5):
        if self.conn_string==None:
            raise Exception("proxy not running. Use the method connect_oblv_proxy to start the proxy.")
        elif self.conn_string=="":
            raise Exception("Either proxy not running or not initiated using syft. Set the conn_string to get started or run the method initiate_connection to initiate the proxy connection")
        depl = self.oblv_client.deployment_info(self.deployment_id)
        if depl.is_deleted==True:
            raise Exception("User cannot connect to this deployment, as it is no longer available.")
        req = requests.post(self.conn_string + "/tensor/publish/request", json={"dataset_id": dataset_id, "sigma": sigma}, headers={"x-oblv-user-name":"vinal", "x-oblv-user-role": "user"})
        if req.status_code==401:
            raise OblvUnAuthorizedError()
        elif req.status_code == 400:
            raise OblvEnclaveError(req.json()["detail"])
        elif req.status_code==422:
            print(req.text)
            #ToDo - Update here
            return "Failed"
        elif req.status_code!=200:
            raise OblvEnclaveError("Request to publish dataset failed with status {}".format(req.status_code))
        return req.json() ##This is the publish_request_id

    def check_publish_request_status(self,publish_request_id):
        if self.conn_string==None:
            raise Exception("proxy not running. Use the method connect_oblv_proxy to start the proxy.")
        elif self.conn_string=="":
            raise Exception("Either proxy not running or not initiated using syft. Set the conn_string to get started or run the method initiate_connection to initiate the proxy connection")
        depl = self.oblv_client.deployment_info(self.deployment_id)
        if depl.is_deleted==True:
            raise Exception("User cannot connect to this deployment, as it is no longer available.")
        req = requests.get(self.conn_string + "/tensor/publish/result_ready?publish_request_id="+publish_request_id
                            , headers={"x-oblv-user-name":"vinal", "x-oblv-user-role": "user"})
        if req.status_code==401:
            raise OblvUnAuthorizedError()
        elif req.status_code == 400:
            raise OblvEnclaveError(req.json()["detail"])
        elif req.status_code==422:
            print(req.text)
            #ToDo - Update here
            return "Failed"
        elif req.status_code!=200:
            raise OblvEnclaveError("Request to publish dataset failed with status {}".format(req.status_code))
        else:
            result = req.json()
            print(result)
            print(type(result))
            if type(result)==str:
                for o in self.client:
                    o.oblv.publish_budget(deployment_id=self.deployment_id,publish_request_id=publish_request_id,client=self.oblv_client)
                print("Not yet Ready")
            else:
                for o in self.client:
                    o.oblv.publish_request_budget_deduction(deployment_id=self.deployment_id,publish_request_id=publish_request_id,client=self.oblv_client,budget_to_deduct=result)
                print("Result is ready") ##This is the publish_request_id

    def fetch_result(self, publish_request_id):
        if self.conn_string==None:
            raise Exception("proxy not running. Use the method connect_oblv_proxy to start the proxy.")
        elif self.conn_string=="":
            raise Exception("Either proxy not running or not initiated using syft. Set the conn_string to get started or run the method initiate_connection to initiate the proxy connection")
        depl = self.oblv_client.deployment_info(self.deployment_id)
        if depl.is_deleted==True:
            raise Exception("User cannot connect to this deployment, as it is no longer available.")
        req = requests.get(self.conn_string + "/tensor/publish/result?request_id="+publish_request_id
                            , headers={"x-oblv-user-name":"vinal", "x-oblv-user-role": "user"})
        if req.status_code==401:
            raise OblvUnAuthorizedError()
        elif req.status_code == 400:
            raise OblvEnclaveError(req.json()["detail"])
        elif req.status_code==422:
            print(req.text)
            #ToDo - Update here
            return "Failed"
        elif req.status_code!=200:
            raise OblvEnclaveError("Request to publish dataset failed with status {}".format(req.status_code))
        return req.json()

    def close_connection(self):
        if self.pid!=None:
            os.kill(self.pid, SIGTERM)
        