# stdlib
from enum import Enum
import json
import os
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ...client.connection import NodeConnection
from ...serde.serializable import serializable
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ..response import SyftError
from ..response import SyftSuccess
from .headscale_client import HeadscaleClient
from .headscale_client import HeadscaleRoutes
from .vpn import BaseVPNClient
from .vpn import VPNClientConnection
from .vpn import VPNRoutes


@serializable()
class TailscaleRoutes(VPNRoutes):
    SERVER_UP = "/commands/up"
    SERVER_DOWN = "/commands/down"
    SERVER_STATUS = "/commands/status"


@serializable()
class ConnectionType(Enum):
    RELAY = "relay"
    DIRECT = "direct"


@serializable()
class ConnectionStatus(Enum):
    ACTIVE = "active"
    IDLE = "idle"


@serializable()
class TailscaleState(Enum):
    RUNNING = "Running"
    STOPPED = "Stopped"


@serializable()
class TailscalePeer(SyftObject):
    __canonical_name__ = "TailscalePeer"
    __version__ = SYFT_OBJECT_VERSION_1

    ip: str
    hostname: str
    network: Optional[str]
    os: Optional[str]
    connection_type: str
    connection_status: str
    is_online: bool


@serializable()
class TailscaleStatus(SyftObject):
    __canonical_name__ = "TailscaleStatus"
    __version__ = SYFT_OBJECT_VERSION_1

    state: str
    peers: Optional[List[TailscalePeer]] = []
    host: TailscalePeer


@serializable()
class TailscaleClient(BaseVPNClient):
    connection: NodeConnection
    api_key: str

    def __init__(self, connection: NodeConnection, api_key: str) -> None:
        self.connection = connection
        self.api_key = api_key

    @staticmethod
    def _extract_host_and_peer(status_dict: Dict) -> Dict:
        def extract_peer_info(peer: Dict) -> Dict:
            info = dict()
            info["hostname"] = peer["HostName"]
            info["os"] = peer["OS"]
            info["ip"] = peer["TailscaleIPs"][0] if peer["TailscaleIPs"] else ""
            info["is_online"] = peer["Online"]
            info["connection_status"] = (
                ConnectionStatus.ACTIVE.value
                if peer["Active"]
                else ConnectionStatus.IDLE.value
            )
            info["connection_type"] = (
                ConnectionType.DIRECT.value
                if peer["CurAddr"]
                else ConnectionType.RELAY.value
            )

            return info

        host_info = extract_peer_info(status_dict["Self"])

        state = status_dict["BackendState"]
        peers = []
        if status_dict["Peer"] is not None:
            for _peer in status_dict["Peer"].values():
                peer_info = extract_peer_info(peer=_peer)
                peers.append(peer_info)

        result = {"state": state, "host": host_info, "peers": peers}

        return result

    def status(self) -> Union[SyftError, TailscaleStatus]:
        # Send command to check tailscale status
        result = self.connection.send_command(
            path=self.connection.routes.SERVER_STATUS.value,
            api_key=self.api_key,
        )

        if result.is_err():
            return SyftError(message=result.err())

        # Get report for the command send to server
        command_report = result.ok()

        # Get result for the executed command
        result = self.connection.resolve_report(
            api_key=self.api_key, report=command_report
        )

        if result.is_err():
            return SyftError(message=result.err())

        command_result = result.ok()

        if command_result.error:
            return SyftError(message=result.error)

        # Get tailscale status as dict
        status_dict = json.loads(command_result.report)

        # Extract host and peer info only
        status_info = self._extract_host_and_peer(status_dict=status_dict)

        return TailscaleStatus(**status_info)

    def connect(
        self, headscale_host: str, headscale_auth_token: str
    ) -> Union[SyftSuccess, SyftError]:
        CONNECT_TIMEOUT = 60

        command_args = {
            "args": [
                "-login-server",
                f"{headscale_host}",
                "--reset",
                "--force-reauth",
                "--authkey",
                f"{headscale_auth_token}",
                "--accept-dns=false",
            ],
        }

        result = self.connection.send_command(
            path=self.connection.routes.SERVER_UP.value,
            api_key=self.api_key,
            timeout=CONNECT_TIMEOUT,
            command_args=command_args,
        )

        if result.is_err():
            return SyftError(message=result.err())

        command_report = result.ok()

        result = self.connection.resolve_report(
            api_key=self.api_key, report=command_report
        )

        if result.is_err():
            return SyftError(message=result.err())

        command_result = result.ok()

        if command_result.error:
            return SyftError(message=result.error)

        return SyftSuccess(message="Connection Successful !")

    def disconnect(self):
        DISCONNECT_TIMEOUT = 60

        result = self.connection.send_command(
            path=self.connection.routes.SERVER_DOWN.value,
            api_key=self.api_key,
            timeout=DISCONNECT_TIMEOUT,
        )

        if result.is_err():
            return SyftError(message=result.err())

        command_report = result.ok()

        result = self.connection.resolve_report(
            api_key=self.api_key, report=command_report
        )

        if result.is_err():
            return SyftError(message=result.err())

        command_result = result.ok()

        if command_result.error:
            return SyftError(message=command_result.error)

        return SyftSuccess(message="Disconnected Successfully !")


def get_vpn_client(
    client_type: Type[BaseVPNClient],
) -> Result[BaseVPNClient, str]:
    api_key = os.getenv("STACK_API_KEY")

    url = None

    if client_type is TailscaleClient:
        url = os.getenv("TAILSCALE_URL", "http://proxy:4000")
        routes = TailscaleRoutes
    elif client_type is HeadscaleClient:
        url = os.getenv("HEADSCALE_URL", "http://headscale:4000")
        routes = HeadscaleRoutes

    if api_key is not None and url is not None:
        client = client_type(
            connection=VPNClientConnection(url=url, routes=routes),
            api_key=api_key,
        )
        return Ok(client)

    return Err(f"Cannot create client for: {client_type.__name__}")
