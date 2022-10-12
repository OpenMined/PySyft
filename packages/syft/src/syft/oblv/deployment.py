# stdlib
import json
from typing import Optional

# third party
from oblv import OblvClient
from oblv.models import CreateDeploymentInput
import yaml

# relative
from ..core.node.common.exceptions import OblvKeyNotFoundError
from .contants import *
from .model import Client
from .model import DeploymentClient
from .oblv_proxy import create_oblv_key_pair
from .oblv_proxy import get_oblv_public_key


def create_deployment(client: OblvClient, domain_clients: list, deployment_name: Optional[str] = None, key_name: str = "", connection_port: int = 3032) -> str:
    """Creates a new deployment with predefined codebase
    Args:
        client : Oblivious Client.
        domain_clients: List of domain_clients.
        deployment_name: Unique name for the deployment.
        key_name: User's key to be used for deployment creation.
    Returns:
        resp: Id of the deployment created
    """
    
    if deployment_name == None:
        deployment_name = input("Kindly provide deployment name")
    if key_name == "":
        key_name = input("Please provide your key name")
    try:
        user_public_key = get_oblv_public_key(key_name)
    except FileNotFoundError:
        user_public_key = create_oblv_key_pair(key_name)
    except Exception as e:
        raise Exception(e)
    build_args = {
        "auth": {},
        "users": {
            "domain": [],
            "user": []
        },
        "additional_args": {},
        "infra_reqs": INFRA,
        "runtime_args": ""
    }
    users = []
    result_client=[]
    runtime_args = []
    for k in domain_clients:
        try:
            users.append({"user_name": k.name, "public key": k.oblv.get_key()})
        except OblvKeyNotFoundError:
            print("Oblv public key not found for {}".format(k.name))
            return
        result_client.append(Client(login=k,datasets=[]))
        runtime_args.append({"domain": k.routes[0].connection.base_url.host_or_ip, "port": k.routes[0].connection.base_url.port, "type": "TCP"})
    build_args["runtime_args"] = yaml.dump({"outbound" : runtime_args})
    build_args["users"]["domain"]=users
    profile = client.user_profile()
    users = [{"user_name": profile.oblivious_login, "public key": user_public_key}]
    build_args["users"]["user"]=users
    depl_input = CreateDeploymentInput(REPO_OWNER, REPO_NAME, VCS,
                                  REF, REGION, deployment_name, VISIBILITY, True, [], build_args)
    res = client.create_deployment(depl_input)
    result = DeploymentClient(deployment_id=res.deployment_id, user_key_name=key_name, connection_port=connection_port)
    return result
