# future
from __future__ import annotations

# stdlib
from concurrent import futures
import json
import os
from typing import Any

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
from .client import SyftClient as Client

NETWORK_REGISTRY_URL = (
    "https://raw.githubusercontent.com/OpenMined/NetworkRegistry/main/gateways.json"
)

NETWORK_REGISTRY_REPO = "https://github.com/OpenMined/NetworkRegistry"


def _get_all_networks(network_json: dict, version: str) -> list[dict]:
    return network_json.get(version, {}).get("gateways", [])


class NetworkRegistry:
    def __init__(self) -> None:
        self.all_networks: list[dict] = []

        try:
            network_json = self.load_network_registry_json()
            self.all_networks = _get_all_networks(
                network_json=network_json, version="2.0.0"
            )
        except Exception as e:
            warning(
                f"Failed to get Network Registry, go checkout: {NETWORK_REGISTRY_REPO}. Exception: {e}"
            )

    @staticmethod
    def load_network_registry_json() -> dict:
        try:
            # Get the environment variable
            network_registry_json = os.getenv("NETWORK_REGISTRY_JSON")
            # If the environment variable exists, use it
            if network_registry_json is not None:
                network_json: dict = json.loads(network_registry_json)
            else:
                # Load the network registry from the NETWORK_REGISTRY_URL
                response = requests.get(NETWORK_REGISTRY_URL, timeout=10)  # nosec
                network_json = response.json()

            return network_json

        except Exception as e:
            warning(
                f"Failed to get Network Registry, go checkout: {NETWORK_REGISTRY_REPO}. {e}"
            )
            return {}

    @property
    def online_networks(self) -> list[dict]:
        networks = self.all_networks

        def check_network(network: dict) -> dict[Any, Any] | None:
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

        return [network for network in _online_networks if network is not None]

    def _repr_html_(self) -> str:
        on = self.online_networks
        if len(on) == 0:
            return "(no gateways online - try syft.gateways.all_networks to see offline gateways)"
        return pd.DataFrame(on)._repr_html_()  # type: ignore

    def __repr__(self) -> str:
        on = self.online_networks
        if len(on) == 0:
            return "(no gateways online - try syft.gateways.all_networks to see offline gateways)"
        return pd.DataFrame(on).to_string()

    @staticmethod
    def create_client(network: dict[str, Any]) -> Client:
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

    def __getitem__(self, key: str | int) -> Client:
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
        self.all_networks: list[dict] = []
        self.all_domains: dict[str, NodePeer] = {}
        try:
            network_json = NetworkRegistry.load_network_registry_json()
            self.all_networks = _get_all_networks(
                network_json=network_json, version="2.0.0"
            )
            self._get_all_domains()
        except Exception as e:
            warning(
                f"Failed to get Network Registry, go checkout: {NETWORK_REGISTRY_REPO}. {e}"
            )

    def _get_all_domains(self) -> None:
        for network in self.all_networks:
            network_client = NetworkRegistry.create_client(network)
            domains: list[NodePeer] = network_client.domains.retrieve_nodes()
            for domain in domains:
                self.all_domains[str(domain.id)] = domain

    @property
    def online_networks(self) -> list[dict]:
        networks = self.all_networks

        def check_network(network: dict) -> dict[Any, Any] | None:
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

        return [network for network in _online_networks if network is not None]

    @property
    def online_domains(self) -> list[tuple[NodePeer, NodeMetadataJSON | None]]:
        def check_domain(
            peer: NodePeer,
        ) -> tuple[NodePeer, NodeMetadataJSON | None] | None:
            try:
                guest_client = peer.guest_client
                metadata = guest_client.metadata
                return peer, metadata
            except Exception as e:  # nosec
                print(f"Error in checking domain with exception {e}")
            return None

        networks = self.online_networks

        # We can use a with statement to ensure threads are cleaned up promptly
        with futures.ThreadPoolExecutor(max_workers=20) as executor:
            # map
            _all_online_domains = []
            for network in networks:
                network_client = NetworkRegistry.create_client(network)
                domains: list[NodePeer] = network_client.domains.retrieve_nodes()
                for domain in domains:
                    self.all_domains[str(domain.id)] = domain
                _online_domains = list(
                    executor.map(lambda domain: check_domain(domain), domains)
                )
                _all_online_domains += _online_domains

        return [domain for domain in _all_online_domains if domain is not None]

    def __make_dict__(self) -> list[dict[str, Any]]:
        on = self.online_domains
        domains: list[dict[str, Any]] = []
        for domain, metadata in on:
            domain_dict: dict[str, Any] = {}
            domain_dict["name"] = domain.name
            if metadata is not None:
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
        on: list[dict[str, Any]] = self.__make_dict__()
        if len(on) == 0:
            return "(no domains online - try syft.domains.all_domains to see offline domains)"
        return pd.DataFrame(on)._repr_html_()  # type: ignore

    def __repr__(self) -> str:
        on: list[dict[str, Any]] = self.__make_dict__()
        if len(on) == 0:
            return "(no domains online - try syft.domains.all_domains to see offline domains)"
        return pd.DataFrame(on).to_string()

    def create_client(self, peer: NodePeer) -> Client:
        try:
            return peer.guest_client
        except Exception as e:
            error(f"Failed to login to: {peer}. {e}")
            raise SyftException(f"Failed to login to: {peer}. {e}")

    def __getitem__(self, key: str | int) -> Client:
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
        self.all_enclaves: list[dict] = []
        try:
            response = requests.get(ENCLAVE_REGISTRY_URL)  # nosec
            enclaves_json = response.json()
            self.all_enclaves = enclaves_json["2.0.0"]["enclaves"]
        except Exception as e:
            warning(
                f"Failed to get Enclave Registry, go checkout: {ENCLAVE_REGISTRY_REPO}. {e}"
            )

    @property
    def online_enclaves(self) -> list[dict]:
        enclaves = self.all_enclaves

        def check_enclave(enclave: dict) -> dict[Any, Any] | None:
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
        return pd.DataFrame(on)._repr_html_()  # type: ignore

    def __repr__(self) -> str:
        on = self.online_enclaves
        if len(on) == 0:
            return "(no enclaves online - try syft.enclaves.all_enclaves to see offline enclaves)"
        return pd.DataFrame(on).to_string()

    @staticmethod
    def create_client(enclave: dict[str, Any]) -> Client:
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

    def __getitem__(self, key: str | int) -> Client:
        if isinstance(key, int):
            return self.create_client(enclave=self.online_enclaves[key])
        else:
            on = self.online_enclaves
            for enclave in on:
                if enclave["name"] == key:
                    return self.create_client(enclave=enclave)
        raise KeyError(f"Invalid key: {key} for {on}")
