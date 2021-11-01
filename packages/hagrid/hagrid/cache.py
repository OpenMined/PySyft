# stdlib
import json
import os
from typing import Any

DEFAULT_BRANCH = "0.6.0"

arg_defaults = {
    "repo": "OpenMined/PySyft",
    "branch": DEFAULT_BRANCH,
    "username": "root",
    "auth_type": "key",
    "key_path": "~/.ssh/id_rsa",
    "azure_repo": "OpenMined/PySyft",
    "azure_branch": DEFAULT_BRANCH,
    "azure_username": "azureuser",
    "azure_key_path": "~/.ssh/id_rsa",
    "azure_resource_group": "openmined",
    "azure_location": "westus",
    "azure_size": "Standard_D2s_v3",
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
        except Exception:
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
            super().__getattr__(key)  # type: ignore


arg_cache = ArgCache()
