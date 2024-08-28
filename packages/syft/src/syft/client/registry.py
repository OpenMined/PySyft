# future
from __future__ import annotations

# stdlib
from concurrent import futures
import json
import logging
import os
from typing import Any

# third party
import pandas as pd
import requests

# relative
from ..service.metadata.server_metadata import ServerMetadataJSON
from ..service.network.server_peer import ServerPeer
from ..service.network.server_peer import ServerPeerConnectionStatus
from ..types.errors import SyftException
from ..types.server_url import ServerURL
from ..types.syft_object import SyftObject
from ..util.constants import DEFAULT_TIMEOUT
from .client import SyftClient as Client

logger = logging.getLogger(__name__)
NETWORK_REGISTRY_URL = (
    "https://raw.githubusercontent.com/OpenMined/NetworkRegistry/main/gateways.json"
)

NETWORK_REGISTRY_REPO = "https://github.com/OpenMined/NetworkRegistry"

DATASITE_REGISTRY_URL = (
    "https://raw.githubusercontent.com/OpenMined/NetworkRegistry/main/datasites.json"
)


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
            logger.warning(
                f"Failed to get Network Registry, go checkout: {NETWORK_REGISTRY_REPO}. Exception: {e}"
            )

    @staticmethod
    def load_network_registry_json() -> dict:
        try:
            # Get the environment variable
            network_registry_json: str | None = os.getenv("NETWORK_REGISTRY_JSON")
            # If the environment variable exists, use it
            if network_registry_json is not None:
                network_json: dict = json.loads(network_registry_json)
            else:
                # Load the network registry from the NETWORK_REGISTRY_URL
                response = requests.get(NETWORK_REGISTRY_URL, timeout=30)  # nosec
                response.raise_for_status()  # raise an exception if the HTTP request returns an error
                network_json = response.json()

            return network_json

        except Exception as e:
            logger.warning(
                f"Failed to get Network Registry from {NETWORK_REGISTRY_REPO}. Exception: {e}"
            )
            return {}

    @property
    def online_networks(self) -> list[dict]:
        networks = self.all_networks

        def check_network(network: dict) -> dict[Any, Any] | None:
            url = "http://" + network["host_or_ip"] + ":" + str(network["port"]) + "/"
            try:
                res = requests.get(url, timeout=DEFAULT_TIMEOUT)  # nosec
                online = "This is a Syft Gateway server." in res.text
            except Exception:
                online = False

            # networks without frontend
            if not online:
                try:
                    ping_url = url + "api/v2/"
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
        df = pd.DataFrame(on)
        total_df = pd.DataFrame(
            [
                [
                    f"{len(on)} / {len(self.all_networks)} (online networks / all networks)"
                ]
                + [""] * (len(df.columns) - 1)
            ],
            columns=df.columns,
            index=["Total"],
        )
        df = pd.concat([df, total_df])
        return df._repr_html_()  # type: ignore

    def __repr__(self) -> str:
        on = self.online_networks
        if len(on) == 0:
            return "(no gateways online - try syft.gateways.all_networks to see offline gateways)"
        df = pd.DataFrame(on)
        total_df = pd.DataFrame(
            [
                [
                    f"{len(on)} / {len(self.all_networks)} (online networks / all networks)"
                ]
                + [""] * (len(df.columns) - 1)
            ],
            columns=df.columns,
            index=["Total"],
        )
        df = pd.concat([df, total_df])
        return df.to_string()

    def __len__(self) -> int:
        return len(self.all_networks)

    @staticmethod
    def create_client(network: dict[str, Any]) -> Client:
        # relative
        from ..client.client import connect

        try:
            port = int(network["port"])
            protocol = network["protocol"]
            host_or_ip = network["host_or_ip"]
            server_url = ServerURL(port=port, protocol=protocol, host_or_ip=host_or_ip)
            client = connect(url=str(server_url))
            return client.guest()
        except Exception as e:
            raise SyftException(public_message=f"Failed to login with: {network}. {e}")

    def __getitem__(self, key: str | int) -> Client:
        if isinstance(key, int):
            return self.create_client(network=self.online_networks[key])
        else:
            on = self.online_networks
            for network in on:
                if network["name"] == key:
                    return self.create_client(network=network)
        raise KeyError(f"Invalid key: {key} for {on}")


class Datasite(SyftObject):
    __canonical_name__ = "ServerMetadata"
    # __version__ = SYFT_OBJECT_VERSION_1

    name: str
    host_or_ip: str
    version: str
    protocol: str
    admin_email: str
    website: str
    slack: str
    slack_channel: str

    __attr_searchable__ = [
        "name",
        "host_or_ip",
        "version",
        "port",
        "admin_email",
        "website",
        "slack",
        "slack_channel",
        "protocol",
    ]
    __attr_unique__ = [
        "name",
        "host_or_ip",
        "version",
        "port",
        "admin_email",
        "website",
        "slack",
        "slack_channel",
        "protocol",
    ]
    __repr_attrs__ = [
        "name",
        "host_or_ip",
        "version",
        "port",
        "admin_email",
        "website",
        "slack",
        "slack_channel",
        "protocol",
    ]
    __table_sort_attr__ = "name"


class DatasiteRegistry:
    def __init__(self) -> None:
        self.all_datasites: list[dict] = []
        try:
            response = requests.get(DATASITE_REGISTRY_URL)  # nosec
            datasites_json = response.json()
            self.all_datasites = datasites_json["datasites"]
        except Exception as e:
            logger.warning(
                f"Failed to get Datasite Registry, go checkout: {DATASITE_REGISTRY_URL}. {e}"
            )

    @property
    def online_datasites(self) -> list[dict]:
        datasites = self.all_datasites

        def check_datasite(datasite: dict) -> dict[Any, Any] | None:
            url = "http://" + datasite["host_or_ip"] + ":" + str(datasite["port"]) + "/"
            try:
                res = requests.get(url, timeout=DEFAULT_TIMEOUT)  # nosec
                if "status" in res.json():
                    online = res.json()["status"] == "ok"
                elif "detail" in res.json():
                    online = True
            except Exception:
                online = False
            if online:
                version = datasite.get("version", None)
                # Check if syft version was described in DatasiteRegistry
                # If it's unknown, try to update it to an available version.
                if not version or version == "unknown":
                    # If not defined, try to ask in /syft/version endpoint (supported by 0.7.0)
                    try:
                        version_url = url + "api/v2/metadata"
                        res = requests.get(version_url, timeout=DEFAULT_TIMEOUT)  # nosec
                        if res.status_code == 200:
                            datasite["version"] = res.json()["syft_version"]
                        else:
                            datasite["version"] = "unknown"
                    except Exception:
                        datasite["version"] = "unknown"
                return datasite
            return None

        # We can use a with statement to ensure threads are cleaned up promptly
        with futures.ThreadPoolExecutor(max_workers=20) as executor:
            # map
            _online_datasites = list(
                executor.map(lambda datasite: check_datasite(datasite), datasites)
            )

        online_datasites = [each for each in _online_datasites if each is not None]
        return online_datasites

    def _repr_html_(self) -> str:
        on = self.online_datasites
        if len(on) == 0:
            return "(no gateways online - try syft.gateways.all_networks to see offline gateways)"

        # df = pd.DataFrame(on)
        print(
            "Add your datasite to this list: https://github.com/OpenMined/NetworkRegistry/"
        )
        # return df._repr_html_()  # type: ignore
        return ([Datasite(**ds) for ds in on])._repr_html_()

    @staticmethod
    def create_client(datasite: dict[str, Any]) -> Client:
        # relative
        from .client import connect

        try:
            port = int(datasite["port"])
            protocol = datasite["protocol"]
            host_or_ip = datasite["host_or_ip"]
            server_url = ServerURL(port=port, protocol=protocol, host_or_ip=host_or_ip)
            client = connect(url=str(server_url))
            return client.guest()
        except Exception as e:
            raise SyftException(public_message=f"Failed to login with: {datasite}. {e}")

    def __getitem__(self, key: str | int) -> Client:
        if isinstance(key, int):
            return self.create_client(datasite=self.online_datasites[key])
        else:
            on = self.online_datasites
            for datasite in on:
                if datasite["name"] == key:
                    return self.create_client(datasite=datasite)
        raise KeyError(f"Invalid key: {key} for {on}")


class NetworksOfDatasitesRegistry:
    def __init__(self) -> None:
        self.all_networks: list[dict] = []
        self.all_datasites: dict[str, ServerPeer] = {}
        try:
            network_json = NetworkRegistry.load_network_registry_json()
            self.all_networks = _get_all_networks(
                network_json=network_json, version="2.0.0"
            )
            self._get_all_datasites()
        except Exception as e:
            logger.warning(
                f"Failed to get Network Registry, go checkout: {NETWORK_REGISTRY_REPO}. {e}"
            )

    def _get_all_datasites(self) -> None:
        for network in self.all_networks:
            network_client = NetworkRegistry.create_client(network)
            datasites: list[ServerPeer] = network_client.datasites.retrieve_servers()
            for datasite in datasites:
                self.all_datasites[str(datasite.id)] = datasite

    @property
    def online_networks(self) -> list[dict]:
        networks = self.all_networks

        def check_network(network: dict) -> dict[Any, Any] | None:
            url = "http://" + network["host_or_ip"] + ":" + str(network["port"]) + "/"
            try:
                res = requests.get(url, timeout=DEFAULT_TIMEOUT)
                online = "This is a Syft Gateway server." in res.text
            except Exception:
                online = False

            # networks without frontend
            if not online:
                try:
                    ping_url = url + "api/v2/"
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
    def online_datasites(self) -> list[tuple[ServerPeer, ServerMetadataJSON | None]]:
        networks = self.online_networks

        _all_online_datasites = []
        for network in networks:
            try:
                network_client = NetworkRegistry.create_client(network)
            except Exception as e:
                logger.error(f"Error in creating network client {e}")
                continue

            datasites: list[ServerPeer] = network_client.datasites.retrieve_servers()
            for datasite in datasites:
                self.all_datasites[str(datasite.id)] = datasite

            _all_online_datasites += [
                (datasite, datasite.guest_client.metadata)
                for datasite in datasites
                if datasite.ping_status == ServerPeerConnectionStatus.ACTIVE
            ]

        return [datasite for datasite in _all_online_datasites if datasite is not None]

    def __make_dict__(self) -> list[dict[str, Any]]:
        on = self.online_datasites
        datasites: list[dict[str, Any]] = []
        for datasite, metadata in on:
            datasite_dict: dict[str, Any] = {}
            datasite_dict["name"] = datasite.name
            if metadata is not None:
                datasite_dict["organization"] = metadata.organization
                datasite_dict["version"] = metadata.syft_version
            route = None
            if len(datasite.server_routes) > 0:
                route = datasite.pick_highest_priority_route()
            datasite_dict["host_or_ip"] = route.host_or_ip if route else "-"
            datasite_dict["protocol"] = route.protocol if route else "-"
            datasite_dict["port"] = route.port if route else "-"
            datasite_dict["id"] = datasite.id
            datasites.append(datasite_dict)

        return datasites

    def _repr_html_(self) -> str:
        on: list[dict[str, Any]] = self.__make_dict__()
        if len(on) == 0:
            return "(no datasites online - try syft.datasites.all_datasites to see offline datasites)"
        df = pd.DataFrame(on)
        total_df = pd.DataFrame(
            [
                [
                    f"{len(on)} / {len(self.all_datasites)} (online datasites / all datasites)"
                ]
                + [""] * (len(df.columns) - 1)
            ],
            columns=df.columns,
            index=["Total"],
        )
        df = pd.concat([df, total_df])
        return df._repr_html_()  # type: ignore

    def __repr__(self) -> str:
        on: list[dict[str, Any]] = self.__make_dict__()
        if len(on) == 0:
            return "(no datasites online - try syft.datasites.all_datasites to see offline datasites)"
        df = pd.DataFrame(on)
        total_df = pd.DataFrame(
            [
                [
                    f"{len(on)} / {len(self.all_datasites)} (online datasites / all datasites)"
                ]
                + [""] * (len(df.columns) - 1)
            ],
            columns=df.columns,
            index=["Total"],
        )
        df = pd.concat([df, total_df])
        return df._repr_html_()  # type: ignore

    def create_client(self, peer: ServerPeer) -> Client:
        try:
            return peer.guest_client
        except Exception as e:
            raise SyftException(public_message=f"Failed to login to: {peer}. {e}")

    def __getitem__(self, key: str | int) -> Client:
        if isinstance(key, int):
            return self.create_client(self.online_datasites[key][0])
        else:
            on = self.online_datasites
            count = 0
            for datasite, _ in on:
                if datasite.name == key:
                    return self.create_client(self.online_datasites[count][0])
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
            logger.warning(
                f"Failed to get Enclave Registry, go checkout: {ENCLAVE_REGISTRY_REPO}. {e}"
            )

    @property
    def online_enclaves(self) -> list[dict]:
        enclaves = self.all_enclaves

        def check_enclave(enclave: dict) -> dict[Any, Any] | None:
            url = "http://" + enclave["host_or_ip"] + ":" + str(enclave["port"]) + "/"
            try:
                res = requests.get(url, timeout=DEFAULT_TIMEOUT)  # nosec
                online = "OpenMined Enclave Server Running" in res.text
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

        online_enclaves = [each for each in _online_enclaves if each is not None]
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
            server_url = ServerURL(port=port, protocol=protocol, host_or_ip=host_or_ip)
            client = connect(url=str(server_url))
            return client.guest()
        except Exception as e:
            raise SyftException(public_message=f"Failed to login with: {enclave}. {e}")

    def __getitem__(self, key: str | int) -> Client:
        if isinstance(key, int):
            return self.create_client(enclave=self.online_enclaves[key])
        else:
            on = self.online_enclaves
            for enclave in on:
                if enclave["name"] == key:
                    return self.create_client(enclave=enclave)
        raise KeyError(f"Invalid key: {key} for {on}")
