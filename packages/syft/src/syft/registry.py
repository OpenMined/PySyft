# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Dict
from typing import Union

# third party
import pandas as pd
import requests

# relative
from . import login
from .core.node.common.client import Client
from .logger import error
from .logger import warning

NETWORK_REGISTRY_URL = (
    "https://raw.githubusercontent.com/OpenMined/NetworkRegistry/main/networks.json"
)
NETWORK_REGISTRY_REPO = "https://github.com/OpenMined/NetworkRegistry"


class NetworkRegistry:
    def __init__(self) -> None:
        self.networks = []
        try:
            response = requests.get(NETWORK_REGISTRY_URL)
            network_json = response.json()
            self.networks = network_json["networks"]
        except Exception as e:
            warning(
                f"Failed to get Network Registry, go checkout: {NETWORK_REGISTRY_REPO}. {e}"
            )

    def _repr_html_(self) -> str:
        return pd.DataFrame(self.networks)._repr_html_()

    def create_client(self, network: Dict[str, Any]) -> Client:
        try:
            host_or_ip = network["host_or_ip"]
            port = int(network["port"])
            protocol = network["protocol"]
            return login(url=f"{protocol}://{host_or_ip}", port=port)
        except Exception as e:
            error(f"Failed to login with: {network}. {e}")
            raise e

    def __getitem__(self, key: Union[str, int]) -> Client:
        if isinstance(key, int):
            return self.create_client(network=self.networks[key])
        else:
            for network in self.networks:
                if network["name"] == key:
                    return self.create_client(network=network)
        raise KeyError(f"Invalid key: {key} for {self.networks}")
