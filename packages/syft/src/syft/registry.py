# future
from __future__ import annotations

# stdlib
from concurrent import futures
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
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

        networks = self.all_networks

        def check_network(network: Dict) -> Optional[Dict[Any, Any]]:
            url = "http://" + network["host_or_ip"] + ":" + str(network["port"]) + "/"
            try:
                res = requests.get(url, timeout=0.5)
                online = "This is a PyGrid Network node." in res.text
            except Exception:
                online = False

            # networks without frontend have a /ping route in 0.7.0
            if not online:
                try:
                    url += "ping"
                    res = requests.get(url, timeout=0.5)
                    online = res.status_code == 200
                except Exception:
                    online = False

            if online:
                return network
            return None

        # We can use a with statement to ensure threads are cleaned up promptly
        with futures.ThreadPoolExecutor(max_workers=20) as executor:
            # map
            _online_networks = list(
                executor.map(lambda network: check_network(network), networks)
            )

        online_networks = list()
        for each in _online_networks:
            if each is not None:
                online_networks.append(each)
        return online_networks

    def _repr_html_(self) -> str:
        on = self.online_networks
        if len(on) == 0:
            return "(no networks online - try syft.networks.all_networks to see offline networks)"
        return pd.DataFrame(on)._repr_html_()

    def __repr__(self) -> str:
        on = self.online_networks
        if len(on) == 0:
            return "(no networks online - try syft.networks.all_networks to see offline networks)"
        return pd.DataFrame(on).to_string()

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
