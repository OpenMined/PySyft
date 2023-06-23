# stdlib
import json
import os
from typing import Any

STABLE_BRANCH = "0.8.1"
DEFAULT_BRANCH = "0.8.1"
DEFAULT_REPO = "OpenMined/PySyft"

arg_defaults = {
    "repo": DEFAULT_REPO,
    "branch": STABLE_BRANCH,
    "username": "root",
    "auth_type": "key",
    "key_path": "~/.ssh/id_rsa",
    "azure_repo": DEFAULT_REPO,
    "azure_branch": STABLE_BRANCH,
    "azure_username": "azureuser",
    "azure_key_path": "~/.ssh/id_rsa",
    "azure_resource_group": "openmined",
    "azure_location": "westus",
    "azure_size": "Standard_D4s_v3",
    "gcp_zone": "us-central1-c",
    "gcp_machine_type": "e2-standard-4",
    "gcp_project_id": "",
    "gcp_username": "",
    "gcp_key_path": "~/.ssh/google_compute_engine",
    "gcp_repo": DEFAULT_REPO,
    "gcp_branch": STABLE_BRANCH,
    "install_wizard_complete": False,
    "aws_region": "us-east-1",
    "aws_security_group_name": "openmined_sg",
    "aws_security_group_cidr": "0.0.0.0/0",
    "aws_image_id": "ami-05de688637f3e33ee",  # Ubuntu Server 22.04 LTS (HVM), SSD Volume Type
    "aws_ec2_instance_type": "t2.xlarge",
    "aws_ec2_instance_username": "ubuntu",  # For Ubuntu AMI, the default user name is ubuntu
    "aws_repo": DEFAULT_REPO,
    "aws_branch": STABLE_BRANCH,
}


class ArgCache:
    @staticmethod
    def cache_file_path() -> str:
        dir_path = os.path.expanduser("~/.hagrid")
        os.makedirs(dir_path, exist_ok=True)
        return f"{dir_path}/cache.json"

    def __init__(self) -> None:
        try:
            with open(ArgCache.cache_file_path(), "r") as f:
                self.__cache = json.loads(f.read())
        except Exception:  # nosec
            self.__cache = {}

    def __setitem__(self, key: str, value: Any) -> None:
        self.__cache[key] = value
        with open(ArgCache.cache_file_path(), "w") as f:
            f.write(json.dumps(self.__cache))

    def __getitem__(self, key: str) -> Any:
        if key in self.__cache:
            return self.__cache[key]
        elif key in arg_defaults:
            return arg_defaults[key]
        raise KeyError(f"Can't find key {key} in ArgCache")


arg_cache = ArgCache()
