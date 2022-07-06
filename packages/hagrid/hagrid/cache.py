# stdlib
import json
import os
from typing import Any

STABLE_BRANCH = "0.6.0"
DEFAULT_BRANCH = "dev"

arg_defaults = {
    "repo": "OpenMined/PySyft",
    "branch": STABLE_BRANCH,
    "username": "root",
    "auth_type": "key",
    "key_path": "~/.ssh/id_rsa",
    "azure_repo": "OpenMined/PySyft",
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
    "gcp_repo": "OpenMined/PySyft",
    "gcp_branch": STABLE_BRANCH,
}


class ArgCache:
    @staticmethod
    def cache_file_path() -> str:
        dir_path = os.path.expanduser("~/.hagrid")
        os.makedirs(dir_path, exist_ok=True)
        return f"{dir_path}/cache.json"

    def __init__(self) -> None:
        cache = {}
        try:
            with open(ArgCache.cache_file_path(), "r") as f:
                cache = json.loads(f.read())
        except Exception:  # nosec
            pass
        self.__dict__ = cache

    def __setattr__(self, key: str, value: Any) -> None:
        super(ArgCache, self).__setattr__(key, value)
        if not key.startswith("__"):
            with open(ArgCache.cache_file_path(), "w") as f:
                f.write(json.dumps(self.__dict__))

    def __getattr__(self, key: str) -> Any:
        if key not in self.__dict__ and key in arg_defaults:
            return arg_defaults[key]
        else:
            print(f"Can't find key {key} in ArgCache")
            super().__getattr__(key)  # type: ignore


arg_cache = ArgCache()
