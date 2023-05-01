# third party
import os
from plistlib import UID
from typing import Optional

from result import Ok
from ..network.network_service import NetworkStash
from ...types.syft_object import SYFT_OBJECT_VERSION_1, SyftObject
from sympy import Union

# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...util.telemetry import instrument
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import service_method
from .vpn_stash import VPNStash


class NodePeerVPNKey(SyftObject):
    __canonical_name__ = "NodePeerVPNKey"
    __version__ = SYFT_OBJECT_VERSION_1

    id: Optional[UID]
    auth_key: str
    name: str


@instrument
@serializable()
class VPNService(AbstractService):
    store: DocumentStore
    stash: NetworkStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = VPNStash(store=store)

    @service_method(path="vpn.join", name="join")
    def join_vpn(
        self,
        context: AuthedServiceContext,
    ) -> Union[SyftSuccess, SyftError]:
        """Join a VPN Service"""
        result = self.register(context=context)

        if result.is_err():
            return result

        node_peer = result.ok()

        result = self.disconnect(tailscale_host=TAILSCALE_URL)
        if result.is_err():
            return result

        result = self.connect_with_key(
            tailscale_host=TAILSCALE_URL,
            headscale_host=str(grid_url.with_path("/vpn")),
            vpn_auth_key=node_peer.auth_key,
        )

        if result.is_err():
            return result

        # node_id = node.node.create_or_get_node(  # type: ignore
        #     node_uid=res_json["node_id"],
        #     node_name=res_json["node_name"],
        # )
        self.stash.add_vpn_endpoint(  # type: ignore
            context=context
            host_or_ip=res_json["host_or_ip"],
            vpn_endpoint=str(grid_url.with_path("/vpn")),
            vpn_key=res_json["vpn_auth_key"],
        )

    def connect_with_key(
        self, tailscale_host: str, headscale_host: str, vpn_auth_key: str
    ) -> Tuple[bool, str]:
        try:
            # we need --accept-dns=false because magicDNS replaces /etc/resolv.conf which
            # breaks using tailscale in network_mode with docker compose because the
            # /etc/resolv.conf has the mDNS ip nameserver 127.0.0.11
            data = {
                "args": [
                    "-login-server",
                    f"{headscale_host}",
                    "--reset",
                    "--force-reauth",
                    "--authkey",
                    f"{vpn_auth_key}",
                    "--accept-dns=false",
                ],
                "timeout": 60,
                "force_unique_key": True,
            }

            command_url = f"{tailscale_host}/commands/up"

            headers = {"X-STACK-API-KEY": os.environ.get("STACK_API_KEY", "")}
            resp = requests.post(command_url, json=data, headers=headers)
            report = get_result(json=resp.json())
            report_dict = json.loads(report)

            if int(report_dict["returncode"]) == 0:
                return (True, "")
            else:
                return (False, report_dict.get("report", ""))
        except Exception as e:
            raise e

    def disconnect(self, tailscale_host: str) -> Tuple[bool, str]:
        try:
            # we need --accept-dns=false because magicDNS replaces /etc/resolv.conf which
            # breaks using tailscale in network_mode with docker compose because the
            # /etc/resolv.conf has the mDNS ip nameserver 127.0.0.11
            data = {
                "args": [],
                "timeout": 60,
                "force_unique_key": True,
            }

            command_url = f"{tailscale_host}/commands/down"

            headers = {"X-STACK-API-KEY": os.environ.get("STACK_API_KEY", "")}
            resp = requests.post(command_url, json=data, headers=headers)
            report = get_result(json=resp.json())
            report_dict = json.loads(report)

            if int(report_dict["returncode"]) == 0:
                return (True, "")
            else:
                return (False, report_dict.get("report", ""))
        except Exception as e:
            critical(f"Failed to disconnect VPN. {e}")
            raise e

    @service_method(path="vpn.status", name="status")
    def get_status(
        self,
        context: AuthedServiceContext,
    ) -> Union[SyftSuccess, SyftError]:
        """Join a VPN Service"""
        pass

    def connect_vpn(
        self,
        context: AuthedServiceContext,
    ) -> Union[SyftSuccess, SyftError]:
        """Join a VPN Service"""
        pass

    @staticmethod
    def _generate_key(headscale_host: str):
        data = {"timeout": 5, "force_unique_key": True}

        command_url = f"{headscale_host}/commands/generate_key"
        try:
            headers = {"X-STACK-API-KEY": os.environ.get("STACK_API_KEY", "")}
            resp = requests.post(command_url, json=data, headers=headers)
            report = get_result(json=resp.json())
            result_dict = dict(extract_nested_json(report))
            result = result_dict["key"]

            # check if we got a key
            if len(result) == 48:
                return (True, result)
            else:
                return (False, "")

        except Exception as e:
            print("failed to make request", e)
            return (False, str(e))

    def register(
        self,
        context: AuthedServiceContext,
    ) -> Union[SyftSuccess, SyftError]:
        """Register node to the VPN."""

        result = self._generate_key(headscale_host=HEADSCALE_URL)

        if result.is_err():
            return result

        node_peer = result.ok()

        return Ok(node_peer)

        # result = {"status": "error"}
        # try:
        #     # get the host_or_ip from tailscale
        #     try:
        #         vpn_metadata = get_vpn_status_metadata(node=node)
        #         result["host_or_ip"] = vpn_metadata["host_or_ip"]
        #     except Exception as e:
        #         print(f"failed to get get_vpn_status_metadata. {e}")
        #         result["host_or_ip"] = "100.64.0.1"
        #     result["node_id"] = str(node.target_id.id.no_dash)
        #     result["node_name"] = str(node.name)
        #     result["status"] = str(reply.payload.kwargs.get("status"))
        #     result["vpn_auth_key"] = str(reply.payload.kwargs.get("vpn_auth_key"))
        # except Exception as e:
