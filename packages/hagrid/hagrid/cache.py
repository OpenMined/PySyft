# stdlib
import json
import os
from typing import Any

STABLE_BRANCH = "0.6.0"
DEFAULT_BRANCH = "dev"
RENDERED_DIR = "rendered"
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
