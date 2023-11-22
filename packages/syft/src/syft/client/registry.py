# future
from __future__ import annotations

# stdlib
from concurrent import futures
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from typing import Tuple
from typing import Union

# third party
import pandas as pd
import requests

# relative
from ..service.metadata.node_metadata import NodeMetadataJSON
from ..service.network.network_service import NodePeer
from ..service.response import SyftException
from ..types.grid_url import GridURL
from ..util.constants import DEFAULT_TIMEOUT
from ..util.logger import error
from ..util.logger import warning
from .enclave_client import EnclaveClient

if TYPE_CHECKING:
    # relative
    from .client import Client

NETWORK_REGISTRY_URL = (
    "https://raw.githubusercontent.com/OpenMined/NetworkRegistry/main/gateways.json"
)
NETWORK_REGISTRY_REPO = "https://github.com/OpenMined/NetworkRegistry"


class NetworkRegistry:
    def __init__(self) -> None:
        self.all_networks: List[Dict] = []
        try:
            response = requests.get(NETWORK_REGISTRY_URL)  # nosec
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
                res = requests.get(url, timeout=DEFAULT_TIMEOUT)  # nosec
                online = "This is a PyGrid Network node." in res.text
            except Exception:
                online = False

            # networks without frontend have a /ping route in 0.7.0
            if not online:
                try:
                    ping_url = url + "ping"
                    res = requests.get(ping_url, timeout=DEFAULT_TIMEOUT)  # nosec
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
                        version_url = url + "api/v2/metadata"
                        res = requests.get(version_url, timeout=DEFAULT_TIMEOUT)  # nosec
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

        online_networks = []
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
        from ..client.client import connect

        try:
            port = int(network["port"])
            protocol = network["protocol"]
            host_or_ip = network["host_or_ip"]
            grid_url = GridURL(port=port, protocol=protocol, host_or_ip=host_or_ip)
            client = connect(url=str(grid_url))
            return client.guest()
        except Exception as e:
            error(f"Failed to login with: {network}. {e}")
            raise SyftException(f"Failed to login with: {network}. {e}")

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
            response = requests.get(NETWORK_REGISTRY_URL)  # nosec
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
                res = requests.get(url, timeout=DEFAULT_TIMEOUT)
                online = "This is a PyGrid Network node." in res.text
            except Exception:
                online = False

            # networks without frontend have a /ping route in 0.7.0
            if not online:
                try:
                    ping_url = url + "ping"
                    res = requests.get(ping_url, timeout=DEFAULT_TIMEOUT)
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
                        version_url = url + "api/v2/metadata"
                        res = requests.get(version_url, timeout=DEFAULT_TIMEOUT)
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

        online_networks = []
        for each in _online_networks:
            if each is not None:
                online_networks.append(each)
        return online_networks

    @property
    def online_domains(self) -> List[Tuple[NodePeer, NodeMetadataJSON]]:
        def check_domain(peer: NodePeer) -> Optional[Tuple[NodePeer, NodeMetadataJSON]]:
            try:
                guest_client = peer.guest_client
                metadata = guest_client.metadata
                return peer, metadata
            except Exception:  # nosec
                pass
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
                _online_domains = list(
                    executor.map(lambda domain: check_domain(domain), domains)
                )
                _all_online_domains += _online_domains

        online_domains = []
        for each in _all_online_domains:
            if each is not None:
                online_domains.append(each)
        return online_domains

    def __make_dict__(self) -> List[Dict[str, str]]:
        on = self.online_domains
        domains = []
        domain_dict = {}
        for domain, metadata in on:
            domain_dict["name"] = domain.name
            domain_dict["organization"] = metadata.organization
            domain_dict["version"] = metadata.syft_version
            route = None
            if len(domain.node_routes) > 0:
                route = domain.pick_highest_priority_route()
            domain_dict["host_or_ip"] = route.host_or_ip if route else "-"
            domain_dict["protocol"] = route.protocol if route else "-"
            domain_dict["port"] = route.port if route else "-"
            domain_dict["id"] = domain.id
            domains.append(domain_dict)
        return domains

    def _repr_html_(self) -> str:
        on = self.__make_dict__()
        if len(on) == 0:
            return "(no domains online - try syft.domains.all_domains to see offline domains)"
        return pd.DataFrame(on)._repr_html_()

    def __repr__(self) -> str:
        on = self.__make_dict__()
        if len(on) == 0:
            return "(no domains online - try syft.domains.all_domains to see offline domains)"
        return pd.DataFrame(on).to_string()

    def create_client(self, peer: NodePeer) -> Client:  # type: ignore
        try:
            return peer.guest_client
        except Exception as e:
            error(f"Failed to login to: {peer}. {e}")
            raise SyftException(f"Failed to login to: {peer}. {e}")

    def __getitem__(self, key: Union[str, int]) -> Client:  # type: ignore
        if isinstance(key, int):
            return self.create_client(self.online_domains[key][0])
        else:
            on = self.online_domains
            count = 0
            for domain, _ in on:
                if domain.name == key:
                    return self.create_client(self.online_domains[count][0])
                count += 1
        raise KeyError(f"Invalid key: {key} for {on}")


ENCLAVE_REGISTRY_URL = (
    "https://raw.githubusercontent.com/OpenMined/NetworkRegistry/main/enclaves.json"
)
ENCLAVE_REGISTRY_REPO = "https://github.com/OpenMined/NetworkRegistry"


class EnclaveRegistry:
    def __init__(self) -> None:
        self.all_enclaves: List[Dict] = []
        try:
            response = requests.get(ENCLAVE_REGISTRY_URL)  # nosec
            enclaves_json = response.json()
            self.all_enclaves = enclaves_json["2.0.0"]["enclaves"]
        except Exception as e:
            warning(
                f"Failed to get Enclave Registry, go checkout: {ENCLAVE_REGISTRY_REPO}. {e}"
            )

    @property
    def online_enclaves(self) -> List[Dict]:
        enclaves = self.all_enclaves

        def check_enclave(enclave: Dict) -> Optional[Dict[Any, Any]]:
            url = "http://" + enclave["host_or_ip"] + ":" + str(enclave["port"]) + "/"
            try:
                res = requests.get(url, timeout=DEFAULT_TIMEOUT)  # nosec
                online = "OpenMined Enclave Node Running" in res.text
            except Exception:
                online = False

            if online:
                version = enclave.get("version", None)
                # Check if syft version was described in EnclaveRegistry
                # If it's unknown, try to update it to an available version.
                if not version or version == "unknown":
                    # If not defined, try to ask in /syft/version endpoint (supported by 0.7.0)
                    try:
                        version_url = url + "api/v2/metadata"
                        res = requests.get(version_url, timeout=DEFAULT_TIMEOUT)  # nosec
                        if res.status_code == 200:
                            enclave["version"] = res.json()["syft_version"]
                        else:
                            enclave["version"] = "unknown"
                    except Exception:
                        enclave["version"] = "unknown"
                return enclave
            return None

        # We can use a with statement to ensure threads are cleaned up promptly
        with futures.ThreadPoolExecutor(max_workers=20) as executor:
            # map
            _online_enclaves = list(
                executor.map(lambda enclave: check_enclave(enclave), enclaves)
            )

        online_enclaves = []
        for each in _online_enclaves:
            if each is not None:
                online_enclaves.append(each)
        return online_enclaves

    def _repr_html_(self) -> str:
        on = self.online_enclaves
        if len(on) == 0:
            return "(no enclaves online - try syft.enclaves.all_enclaves to see offline enclaves)"
        return pd.DataFrame(on)._repr_html_()

    def __repr__(self) -> str:
        on = self.online_enclaves
        if len(on) == 0:
            return "(no enclaves online - try syft.enclaves.all_enclaves to see offline enclaves)"
        return pd.DataFrame(on).to_string()

    @staticmethod
    def create_client(enclave: Dict[str, Any]) -> Client:  # type: ignore
        # relative
        from ..client.client import connect

        try:
            port = int(enclave["port"])
            protocol = enclave["protocol"]
            host_or_ip = enclave["host_or_ip"]
            grid_url = GridURL(port=port, protocol=protocol, host_or_ip=host_or_ip)
            client = connect(url=str(grid_url))
            return client.guest()
        except Exception as e:
            error(f"Failed to login with: {enclave}. {e}")
            raise SyftException(f"Failed to login with: {enclave}. {e}")

    def __getitem__(self, key: Union[str, int]) -> EnclaveClient:  # type: ignore
        if isinstance(key, int):
            return self.create_client(enclave=self.online_enclaves[key])
        else:
            on = self.online_enclaves
            for enclave in on:
                if enclave["name"] == key:
                    return self.create_client(enclave=enclave)
        raise KeyError(f"Invalid key: {key} for {on}")
