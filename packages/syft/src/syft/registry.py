# future
from __future__ import annotations

# stdlib
import sys
from typing import Any
from typing import Dict
from typing import List
from typing import Union

# third party
import pandas as pd
import requests

# relative
from . import login
from .core.node.common.client import Client
from .grid import GridURL
from .logger import error
from .logger import warning

NETWORK_REGISTRY_URL = (
    "https://raw.githubusercontent.com/OpenMined/NetworkRegistry/main/networks.json"
)
NETWORK_REGISTRY_REPO = "https://github.com/OpenMined/NetworkRegistry"


class NetworkRegistry:
    def __init__(self) -> None:
        self.networks: List[Dict] = []
        try:
            response = requests.get(NETWORK_REGISTRY_URL)
            network_json = response.json()
            self.all_networks = network_json["networks"]
        except Exception as e:
            warning(
                f"Failed to get Network Registry, go checkout: {NETWORK_REGISTRY_REPO}. {e}"
            )

    @property
    def online_networks(self) -> List[Dict]:
        online_networks = list()

        an = self.all_networks

        for i, network in enumerate(an):
            sys.stdout.write(
                "\rChecking network availability: " + str(i + 1) + " of " + str(len(an))
            )
            url = "http://" + network["host_or_ip"] + ":" + str(network["port"]) + "/"
            try:
                res = requests.get(url, timeout=0.5)
                online = "This is a PyGrid Network node." in res.text
            except Exception:
                online = False

            if online:
                online_networks.append(network)
        sys.stdout.write("\r                                             ")
        return online_networks

    def _repr_html_(self) -> str:
        on = self.online_networks
        if len(on) == 0:
            return "(no networks online - try syft.networks.all_networks to see offline networks)"
        return pd.DataFrame(on)._repr_html_()

    def create_client(self, network: Dict[str, Any]) -> Client:
        try:
            port = int(network["port"])
            protocol = network["protocol"]
            host_or_ip = network["host_or_ip"]
            grid_url = GridURL(port=port, protocol=protocol, host_or_ip=host_or_ip)
            return login(url=str(grid_url), port=port)
        except Exception as e:
            error(f"Failed to login with: {network}. {e}")
            raise e

    def __getitem__(self, key: Union[str, int]) -> Client:
        if isinstance(key, int):
            return self.create_client(network=self.online_networks[key])
        else:
            on = self.online_networks
            for network in on:
                if network["name"] == key:
                    return self.create_client(network=network)
        raise KeyError(f"Invalid key: {key} for {on}")
