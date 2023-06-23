# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

# third party
from oblv_ctl import OblvClient
from oblv_ctl.models import CreateDeploymentInput
import yaml

# relative
from ...util.util import bcolors
from .auth import login
from .constants import INFRA
from .constants import REF
from .constants import REF_TYPE
from .constants import REGION
from .constants import REPO_NAME
from .constants import REPO_OWNER
from .constants import VCS
from .constants import VISIBILITY
from .deployment_client import DeploymentClient
from .exceptions import OblvKeyNotFoundError
from .oblv_proxy import create_oblv_key_pair
from .oblv_proxy import get_oblv_public_key

SUPPORTED_REGION_LIST = ["us-east-1", "us-west-2", "eu-central-1", "eu-west-2"]

SUPPORTED_INFRA = [
    "c5.xlarge",
    "m5.xlarge",
    "r5.xlarge",
    "c5.2xlarge",
    "m5.2xlarge",
    "m5.4xlarge",
]


def create_deployment(
    domain_clients: list,
    deployment_name: Optional[str] = None,
    key_name: Optional[str] = None,
    oblv_client: Optional[OblvClient] = None,
    infra: str = INFRA,
    region: str = REGION,
) -> DeploymentClient:
    """Creates a new deployment with predefined codebase
    Args:
        client : Oblivious Client.
        domain_clients: List of domain_clients.
        deployment_name: Unique name for the deployment.
        key_name: User's key to be used for deployment creation.
        infra: Represent the AWS infrastructure to be used. Default is "m5.2xlarge". The available options are\n
                - "c5.xlarge" {'CPU':4, 'RAM':8, 'Total/hr':0.68}\n
                - "m5.xlarge" {'CPU':4, 'RAM':16, 'Total/hr':0.768}\n
                - "r5.xlarge" {'CPU':4, 'RAM':32, 'Total/hr':1.008}\n
                - "c5.2xlarge" {'CPU':8, 'RAM':16, 'Total/hr':1.36}\n
                - "m5.2xlarge" {'CPU':8, 'RAM':32, 'Total/hr':1.536}\n
                As of now, PySyft only works with RAM >= 32
        region: AWS Region to be deployed in. Default is "us-east-1". The available options are \n
                - "us-east-1" : "US East (N. Virginia)",\n
                - "us-west-2" : "US West (Oregon)",\n
                - "eu-central-1" : "Europe (Frankfurt)",\n
                - "eu-west-2" : "Europe (London)"

    Returns:
        resp: Deployment Client Object
    """
    if not oblv_client:
        oblv_client = login()
    if deployment_name is None:
        deployment_name = input("Kindly provide deployment name")
    if key_name is None:
        key_name = input("Please provide your key name")

    while not SUPPORTED_INFRA.__contains__(infra):
        infra = input(f"Provide infra from one of the following - {SUPPORTED_INFRA}")

    while not SUPPORTED_REGION_LIST.__contains__(region):
        region = input(
            f"Provide region from one of the following - {SUPPORTED_REGION_LIST}"
        )

    try:
        user_public_key = get_oblv_public_key(key_name)
    except FileNotFoundError:
        user_public_key = create_oblv_key_pair(key_name)
        print(
            bcolors.green(bcolors.bold("Created"))
            + f" a new public/private key pair with key_name: {key_name}"
        )
    except Exception as e:
        raise Exception(e)
    build_args: Dict[str, Any] = {
        "auth": {},
        "users": {"domain": [], "user": []},
        "additional_args": {},
        "infra_reqs": infra,
        "runtime_args": "",
    }
    users = []
    runtime_args: List[str] = []
    for domain_client in domain_clients:
        try:
            users.append(
                {
                    "user_name": domain_client.name,
                    "public key": domain_client.api.services.oblv.get_public_key(),
                }
            )
        except OblvKeyNotFoundError:
            raise OblvKeyNotFoundError(
                f"Oblv Public Key not found for {domain_client.name}"
            )

    build_args["runtime_args"] = yaml.dump({"outbound": runtime_args})
    build_args["users"]["domain"] = users
    profile = oblv_client.user_profile()
    users = [{"user_name": profile.oblivious_login, "public key": user_public_key}]
    build_args["users"]["user"] = users
    depl_input = CreateDeploymentInput(
        owner=REPO_OWNER,
        repo=REPO_NAME,
        account_type=VCS,
        ref=REF,
        ref_type=REF_TYPE,
        region_name=region,
        deployment_name=deployment_name,
        visibility=VISIBILITY,
        is_dev_env=True,
        tags=[],
        build_args=build_args,
    )
    # By default the deployment is in PROD mode
    res = oblv_client.create_deployment(depl_input)
    result = DeploymentClient(
        deployment_id=res.deployment_id,
        oblv_client=oblv_client,
        domain_clients=domain_clients,
        key_name=key_name,
    )
    return result
