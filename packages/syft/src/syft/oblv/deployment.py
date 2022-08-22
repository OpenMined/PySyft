from typing import Optional
from oblv import OblvClient
from oblv.models import CreateDeploymentInput
from ..core.node.common.exceptions import OblvKeyNotFoundError
from .contants import *


def create_deployment(client: OblvClient, domain_clients: list, deployment_name: Optional[str] = None) -> str:
    """Creates a new deployment with predefined codebase
    Args:
        client : Oblivious Client.
        domain_clients: List of domain_clients
    Returns:
        resp: Id of the deployment created
    """
    if deployment_name == None:
        deployment_name = input("Kindly provide deployment name")
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
    for k in domain_clients:
        try:
            users.append({"user_name": k.name, "public key": k.oblv.get_key()})
        except OblvKeyNotFoundError:
            print("Oblv public key not found for {}".format(k.name))
            return
    
    build_args["users"]["domain"]=users
    profile = client.user_profile()
    if profile.public_key == "":
        profile.public_key = input("Please provide your public key")
    users = [{"user_name": profile.oblivious_login, "public key": profile.public_key}]
    build_args["users"]["user"]=users
    depl_input = CreateDeploymentInput(REPO_OWNER, REPO_NAME, VCS,
                                  REF, REGION, deployment_name, VISIBILITY, True, [], build_args)
    res = client.create_deployment(depl_input)
    return res.deployemnt_id
