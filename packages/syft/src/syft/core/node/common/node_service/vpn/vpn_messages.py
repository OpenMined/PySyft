# future
from __future__ import annotations

# stdlib
import json
import os
import re
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
from nacl.signing import VerifyKey
import requests
from typing_extensions import final

# relative
from ......grid import GridURL
from ......logger import critical
from ......util import verify_tls
from .....common.serde.serializable import serializable
from ....abstract.node import AbstractNode
from ..generic_payload.messages import GenericPayloadMessage
from ..generic_payload.messages import GenericPayloadMessageWithReply
from ..generic_payload.messages import GenericPayloadReplyMessage

TAILSCALE_HOST = "proxy"
TAILSCALE_PORT = 4000
TAILSCALE_URL = f"http://{TAILSCALE_HOST}:{TAILSCALE_PORT}"

HEADSCALE_HOST = "headscale"
HEADSCALE_PORT = 4000
HEADSCALE_URL = f"http://{HEADSCALE_HOST}:{HEADSCALE_PORT}"


def grid_url_from_kwargs(kwargs: Dict[str, Any]) -> GridURL:
    try:
        if "host_or_ip" in kwargs:
            # old way to send these messages was with host_or_ip
            return GridURL.from_url(str(kwargs["host_or_ip"]))
        elif "grid_url" in kwargs:
            # new way is with grid_url
            return kwargs["grid_url"]
        else:
            raise Exception("kwargs missing host_or_ip or grid_url")
    except Exception as e:
        print(f"Failed to get grid_url from kwargs: {kwargs}. {e}")
        raise e


@serializable(recursive_serde=True)
@final
class VPNConnectMessage(GenericPayloadMessage):
    ...


@serializable(recursive_serde=True)
@final
class VPNConnectReplyMessage(GenericPayloadReplyMessage):
    ...


@serializable(recursive_serde=True)
@final
class VPNConnectMessageWithReply(GenericPayloadMessageWithReply):
    message_type = VPNConnectMessage
    message_reply_type = VPNConnectReplyMessage

    def run(
        self, node: AbstractNode, verify_key: Optional[VerifyKey] = None
    ) -> Dict[str, Any]:
        try:
            grid_url = grid_url_from_kwargs(self.kwargs)
            grid_url = grid_url.with_path("/vpn")
            vpn_auth_key = str(self.kwargs["vpn_auth_key"])

            status, error = disconnect(tailscale_host=TAILSCALE_URL)
            if not status:
                print("Failed to run tailscale down first", error)

            status, error = connect_with_key(
                tailscale_host=TAILSCALE_URL,
                headscale_host=str(grid_url),
                vpn_auth_key=vpn_auth_key,
            )
            if status:
                return {"status": "ok"}
            else:
                print("connect with key failed", error)
                return {"status": "error"}
        except Exception as e:
            print(f"Failed to run {type(self)}", self.kwargs, e)
            return {"status": "error"}


def disconnect(tailscale_host: str) -> Tuple[bool, str]:
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


def connect_with_key(
    tailscale_host: str, headscale_host: str, vpn_auth_key: str
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
        critical(f"Failed to connect to VPN. {e}")
        raise e


@serializable(recursive_serde=True)
@final
class VPNJoinMessage(GenericPayloadMessage):
    ...


@serializable(recursive_serde=True)
@final
class VPNJoinReplyMessage(GenericPayloadReplyMessage):
    ...


@serializable(recursive_serde=True)
@final
class VPNJoinMessageWithReply(GenericPayloadMessageWithReply):
    message_type = VPNJoinMessage
    message_reply_type = VPNJoinReplyMessage

    def run(
        self, node: AbstractNode, verify_key: Optional[VerifyKey] = None
    ) -> Dict[str, Any]:
        try:
            # can't import Network due to circular imports
            if type(node).__name__ == "Network":
                # networks can't join other networks yet
                raise Exception(
                    "Network cant join another Network, try VPNJoinSelfMessageWithReply"
                )

            # we are running inside the container so we should change the host to
            # what ever will suit the environment with as_container_host
            grid_url = grid_url_from_kwargs(self.kwargs).as_container_host(
                container_host=node.settings.CONTAINER_HOST
            )
            res = requests.post(
                str(grid_url.with_path("/api/v1/vpn/register")), verify=verify_tls()
            )
            res_json = res.json()
            if "vpn_auth_key" not in res_json:
                print("Registration failed", res)
                return {"status": "error"}

            status, error = disconnect(tailscale_host=TAILSCALE_URL)
            if not status:
                print("Failed to run tailscale down first", error)

            status, error = connect_with_key(
                tailscale_host=TAILSCALE_URL,
                headscale_host=str(grid_url.with_path("/vpn")),
                vpn_auth_key=res_json["vpn_auth_key"],
            )

            if status:
                node_id = node.node.create_or_get_node(  # type: ignore
                    node_uid=res_json["node_id"],
                    node_name=res_json["node_name"],
                )
                node.node_route.update_route_for_node(  # type: ignore
                    node_id=node_id,
                    host_or_ip=res_json["host_or_ip"],
                    vpn_endpoint=str(grid_url.with_path("/vpn")),
                    vpn_key=res_json["vpn_auth_key"],
                )
                return {"status": "ok"}
            else:
                print("connect with key failed", error)
                return {"status": "error"}
        except Exception as e:
            print(f"Failed to run {type(self)}", self.kwargs, e)
            return {"status": "error"}


@serializable(recursive_serde=True)
@final
class VPNJoinSelfMessage(GenericPayloadMessage):
    ...


@serializable(recursive_serde=True)
@final
class VPNJoinSelfReplyMessage(GenericPayloadReplyMessage):
    ...


@serializable(recursive_serde=True)
@final
class VPNJoinSelfMessageWithReply(GenericPayloadMessageWithReply):
    message_type = VPNJoinSelfMessage
    message_reply_type = VPNJoinSelfReplyMessage

    def run(
        self, node: AbstractNode, verify_key: Optional[VerifyKey] = None
    ) -> Dict[str, Any]:
        try:
            # can't import Domaiun due to circular imports
            if type(node).__name__ == "Domain":
                # networks can't join other networks yet
                raise Exception(
                    "Domains cant join themselves, try VPNJoinMessageWithReply"
                )

            # get status
            up_status, _, _ = get_status(tailscale_host=TAILSCALE_URL)
            if up_status:
                print("Already connected!")
                return {"status": "ok"}
            print("Connecting...")

            key_status, vpn_auth_key = generate_key(headscale_host=HEADSCALE_URL)

            if not key_status:
                raise Exception("Failed to generate key for joining VPN")

            status, error = disconnect(tailscale_host=TAILSCALE_URL)
            if not status:
                print("Failed to run tailscale down first", error)

            connect_status, error = connect_with_key(
                tailscale_host=TAILSCALE_URL,
                headscale_host=f"http://{HEADSCALE_HOST}:8080",
                vpn_auth_key=vpn_auth_key,
            )

            if connect_status:
                print("Connected successfully")
                return {"status": "ok"}
            else:
                print("connect with key failed", error)
                return {"status": "error"}
        except Exception as e:
            print(f"Failed to run {type(self)}", self.kwargs, e)
            return {"status": "error"}


@serializable(recursive_serde=True)
@final
class VPNRegisterMessage(GenericPayloadMessage):
    ...


@serializable(recursive_serde=True)
@final
class VPNRegisterReplyMessage(GenericPayloadReplyMessage):
    ...


@serializable(recursive_serde=True)
@final
class VPNRegisterMessageWithReply(GenericPayloadMessageWithReply):
    message_type = VPNRegisterMessage
    message_reply_type = VPNRegisterReplyMessage

    def run(self, node: AbstractNode, verify_key: Optional[VerifyKey] = None) -> Any:
        try:
            status, vpn_auth_key = generate_key(headscale_host=HEADSCALE_URL)

            if status:
                return {"status": "ok", "vpn_auth_key": vpn_auth_key}
            else:
                print("register and create vpn_auth_key failed")
                return {"status": "error"}
        except Exception as e:
            print(f"Failed to run {type(self)}", self.kwargs, e)
            return {"status": "error"}


# the response contains a dict with a key called "report" which contains unescaped json
def extract_nested_json(nested_json: str) -> Union[Dict, List]:
    matcher = r'"report":"(.+)\\n"'
    parts = re.findall(matcher, nested_json)
    report_json = parts[0].replace("\\n", "").replace("\\t", "").replace('\\"', '"')
    return json.loads(report_json)


def generate_key(headscale_host: str) -> Tuple[bool, str]:
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


@serializable(recursive_serde=True)
@final
class VPNStatusMessage(GenericPayloadMessage):
    ...


@serializable(recursive_serde=True)
@final
class VPNStatusReplyMessage(GenericPayloadReplyMessage):
    ...


@serializable(recursive_serde=True)
@final
class VPNStatusMessageWithReply(GenericPayloadMessageWithReply):
    message_type = VPNStatusMessage
    message_reply_type = VPNStatusReplyMessage

    def run(self, node: AbstractNode, verify_key: Optional[VerifyKey] = None) -> Any:
        try:
            up, host, peers = get_status(tailscale_host=TAILSCALE_URL)
            return {"status": "ok", "connected": up, "host": host, "peers": peers}
        except Exception as e:
            print(f"Failed to run {type(self)}", self.kwargs, e)
            return {"status": "error"}


def clean_status_output(
    input: str,
) -> Tuple[bool, Dict[str, str], List[Dict[str, str]]]:
    # example input
    """
    # Health check:
    #     - dns: rename /etc/resolv.conf /etc/resolv.pre-tailscale-backup.conf: device or resource busy

    100.64.0.1      test_domain_1        omnet        linux   -
    100.64.0.2      test_network_1       omnet        linux   active; relay "syd", tx 1188 rx 1040
    """
    up = False
    peers: List[Dict[str, str]] = []
    host: Dict[str, str] = {}
    if "Tailscale is stopped." in input:
        return up, host, peers
    elif "unexpected state: NoState" in input:
        return up, host, peers

    count = 0
    for line in str(input).split("\n"):
        matches = re.match(r"^\d.+", line)
        if matches is not None:
            try:
                stat_parts = re.split(r"(\s+)", matches.string)
                entry = {}
                entry["ip"] = stat_parts[0]
                entry["hostname"] = stat_parts[2]
                entry["network"] = stat_parts[4]
                entry["os"] = stat_parts[6]
                connection_info_parts = matches.string.split(entry["os"])
                entry["connection_info"] = "n/a"
                connection_info = ""
                if len(connection_info_parts) > 1:
                    connection_info = connection_info_parts[1].strip()
                    entry["connection_info"] = connection_info

                entry["connection_status"] = "n/a"
                if "active" in connection_info:
                    entry["connection_status"] = "active"

                if "idle" in connection_info:
                    entry["connection_status"] = "idle"

                entry["connection_type"] = "n/a"
                if "relay" in connection_info:
                    entry["connection_type"] = "relay"

                if "direct" in connection_info:
                    entry["connection_type"] = "direct"

                if count == 0:
                    host = entry
                    count += 1
                    up = True
                else:
                    peers.append(entry)

            except Exception as e:
                print("Error parsing tailscale status output", e)
                pass

    return up, host, peers


def get_status(
    tailscale_host: str,
) -> Tuple[bool, Dict[str, str], List[Dict[str, str]]]:
    data = {"timeout": 5, "force_unique_key": True}
    command_url = f"{tailscale_host}/commands/status"
    host: Dict[str, str] = {}
    peers: List[Dict[str, str]] = []
    connected = False
    try:
        headers = {"X-STACK-API-KEY": os.environ.get("STACK_API_KEY", "")}
        resp = requests.post(command_url, json=data, headers=headers)
        report = get_result(json=resp.json())
        cmd_output = json.loads(report)["report"]
        connected, host, peers = clean_status_output(input=cmd_output)
    except Exception as e:
        print("failed to make request", e)
    return connected, host, peers


def get_network_url(tailscale_host: str = TAILSCALE_URL) -> Union[None, GridURL]:
    data = {"timeout": 5, "force_unique_key": True}
    command_url = f"{tailscale_host}/commands/debug"
    try:
        headers = {"X-STACK-API-KEY": os.environ.get("STACK_API_KEY", "")}
        resp = requests.post(command_url, json=data, headers=headers)
        report = get_result(json=resp.json())
        network_url = GridURL.from_url(
            json.loads(json.loads(report)["report"])["ControlURL"][:-4]
        )
        return network_url
    except Exception as e:
        print("failed to make request", e)
    return None


def get_result(json: Dict) -> str:
    result_url = json.get("result_url", "")
    headers = {"X-STACK-API-KEY": os.environ.get("STACK_API_KEY", "")}
    tries = 0
    limit = 5
    try:
        while True:
            print("Polling API Result", tries)
            result = requests.get(result_url, headers=headers)
            if '"status":"running"' in result.text:
                time.sleep(1)
                tries += 1
                if tries > limit:
                    break
                continue
            else:
                return str(result.text)
    except Exception as e:
        print("Failed to get result", json, e)
    return "{}"
