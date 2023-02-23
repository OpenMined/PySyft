# future
from __future__ import annotations

# stdlib
from concurrent import futures
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union

# third party
import pandas as pd
import requests

# relative
from .grid import GridURL
from .logger import error
from .logger import warning

if TYPE_CHECKING:
    # relative
    from .core.node.common.client import Client

NETWORK_REGISTRY_URL = (
    "https://raw.githubusercontent.com/OpenMined/NetworkRegistry/main/gateways.json"
)
NETWORK_REGISTRY_REPO = "https://github.com/OpenMined/NetworkRegistry"


class NetworkRegistry:
    def __init__(self) -> None:
        self.all_networks: List[Dict] = []
        try:
            response = requests.get(NETWORK_REGISTRY_URL)
            network_json = response.json()
            self.all_networks = network_json["2.0.0"]["gateways"]
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
                    ping_url = url + "ping"
                    res = requests.get(ping_url, timeout=0.5)
                    online = res.status_code == 200
                except Exception:
                    online = False

            if online:
                version = network.get("version", None)
                # Check if syft version was described in NetworkRegistry
                # If it's unknown, try to update it to an available version.
                if not version or version == "unknown":
                    # If not defined, try to ask in /syft/version endpoint (supported by 0.7.0)
                    try:
                        version_url = url + "api/v1/new/metadata"
                        res = requests.get(version_url, timeout=0.5)
                        if res.status_code == 200:
                            network["version"] = res.json()["syft_version"]
                        else:
                            network["version"] = "unknown"
                    except Exception:
                        network["version"] = "unknown"
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
            return "(no gateways online - try syft.gateways.all_gateways to see offline gateways)"
        return pd.DataFrame(on)._repr_html_()

    def __repr__(self) -> str:
        on = self.online_networks
        if len(on) == 0:
            return "(no gateways online - try syft.gateways.all_gateways to see offline gateways)"
        return pd.DataFrame(on).to_string()

    @staticmethod
    def create_client(network: Dict[str, Any]) -> Client:  # type: ignore
        # relative
        from .core.node.new.client import connect

        try:
            port = int(network["port"])
            protocol = network["protocol"]
            host_or_ip = network["host_or_ip"]
            grid_url = GridURL(port=port, protocol=protocol, host_or_ip=host_or_ip)
            client = connect(url=str(grid_url))
            return client.guest()
        except Exception as e:
            error(f"Failed to login with: {network}. {e}")
            raise e

    def __getitem__(self, key: Union[str, int]) -> Client:  # type: ignore
        if isinstance(key, int):
            return self.create_client(network=self.online_networks[key])
        else:
            on = self.online_networks
            for network in on:
                if network["name"] == key:
                    return self.create_client(network=network)
        raise KeyError(f"Invalid key: {key} for {on}")


class DomainRegistry:
    def __init__(self) -> None:
        self.all_networks: List[Dict] = []
        self.all_domains: List = []
        try:
            response = requests.get(NETWORK_REGISTRY_URL)
            network_json = response.json()
            self.all_networks = network_json["2.0.0"]["gateways"]
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
                    ping_url = url + "ping"
                    res = requests.get(ping_url, timeout=0.5)
                    online = res.status_code == 200
                except Exception:
                    online = False

            if online:
                version = network.get("version", None)
                # Check if syft version was described in NetworkRegistry
                # If it's unknown, try to update it to an available version.
                if not version or version == "unknown":
                    # If not defined, try to ask in /syft/version endpoint (supported by 0.7.0)
                    try:
                        version_url = url + "api/v1/new/metadata"
                        res = requests.get(version_url, timeout=0.5)
                        if res.status_code == 200:
                            network["version"] = res.json()["syft_version"]
                        else:
                            network["version"] = "unknown"
                    except Exception:
                        network["version"] = "unknown"
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

    @property
    def online_domains(self) -> List[Dict]:
        def check_domain(domain: Dict) -> Optional[Dict[Any, Any]]:
            url = "http://" + domain["host_or_ip"] + ":" + str(domain["port"]) + "/"
            try:
                res = requests.get(url, timeout=0.5)
                online = "This is a PyGrid Network node." in res.text
            except Exception:
                online = False

            # networks without frontend have a /ping route in 0.7.0
            if not online:
                try:
                    ping_url = url + "ping"
                    res = requests.get(ping_url, timeout=0.5)
                    online = res.status_code == 200
                except Exception:
                    online = False

            if online:
                version = domain.get("syft_version", None)
                # Check if syft version was described in NetworkRegistry
                # If it's unknown, try to update it to an available version.
                if not version or version == "unknown":
                    # If not defined, try to ask in /syft/version endpoint (supported by 0.7.0)
                    try:
                        version_url = url + "api/v1/new/metadata"
                        res = requests.get(version_url, timeout=0.5)
                        if res.status_code == 200:
                            network["version"] = res.json()["syft_version"]
                        else:
                            network["version"] = "unknown"
                    except Exception:
                        network["version"] = "unknown"
                return network
            return None

        networks = self.online_networks

        self.all_domains = []
        # We can use a with statement to ensure threads are cleaned up promptly
        with futures.ThreadPoolExecutor(max_workers=20) as executor:
            # map
            _all_online_domains = []
            for network in networks:
                network_client = NetworkRegistry.create_client(network)
                domains = network_client.domains
                self.all_domains += domains
                print("k", network)
                print("domaisn", domains)
                _online_domains = list(
                    executor.map(lambda domain: check_domain(domain), domains)
                )
                _all_online_domains += _online_domains

        online_domains = list()
        for each in _all_online_domains:
            if each is not None:
                online_domains.append(each)
        return online_domains

    def _repr_html_(self) -> str:
        on = self.online_domains
        if len(on) == 0:
            return "(no domains online - try syft.domains.all_domains to see offline domains)"
        return pd.DataFrame(on)._repr_html_()

    def __repr__(self) -> str:
        on = self.online_domains
        if len(on) == 0:
            return "(no domains online - try syft.domains.all_domains to see offline domains)"
        return pd.DataFrame(on).to_string()

    def create_client(self, domain: Dict[str, Any]) -> Client:  # type: ignore
        # relative
        from .core.node.new.client import connect

        try:
            port = int(domain["port"])
            protocol = domain["protocol"]
            host_or_ip = domain["host_or_ip"]
            grid_url = GridURL(port=port, protocol=protocol, host_or_ip=host_or_ip)
            client = connect(url=str(grid_url))
            return client.guest()
        except Exception as e:
            error(f"Failed to login with: {domain}. {e}")
            raise e

    def __getitem__(self, key: Union[str, int]) -> Client:  # type: ignore
        if isinstance(key, int):
            return self.create_client(network=self.online_networks[key])
        else:
            on = self.online_networks
            for network in on:
                if network["name"] == key:
                    return self.create_client(network=network)
        raise KeyError(f"Invalid key: {key} for {on}")
