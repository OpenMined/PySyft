# future
from __future__ import annotations

# stdlib
import json
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
from .....common.serde.serializable import serializable
from ....abstract.node import AbstractNode
from ..generic_payload.messages import GenericPayloadMessage
from ..generic_payload.messages import GenericPayloadMessageWithReply
from ..generic_payload.messages import GenericPayloadReplyMessage


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
            host_or_ip = str(self.kwargs["host_or_ip"])
            if not host_or_ip.startswith("http"):
                host_or_ip = f"http://{host_or_ip}/vpn"

            vpn_auth_key = str(self.kwargs["vpn_auth_key"])

            status, error = connect_with_key(
                tailscale_host="http://tailscale:4000",
                headscale_host=host_or_ip,
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


def connect_with_key(
    tailscale_host: str, headscale_host: str, vpn_auth_key: str
) -> Tuple[bool, str]:
    data = {
        "args": [
            "-login-server",
            f"{headscale_host}",
            "--reset",
            "--force-reauth",
            "--authkey",
            f"{vpn_auth_key}",
        ],
        "timeout": 60,
    }
    command_url = f"{tailscale_host}/commands/up"

    resp = requests.post(command_url, json=data)
    report = get_result(json=resp.json())
    report_dict = json.loads(report)

    if int(report_dict["returncode"]) == 0:
        return (True, "")
    else:
        return (False, report_dict.get("report", ""))


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
            host_or_ip = str(self.kwargs["host_or_ip"])
            if not host_or_ip.startswith("http"):
                host_or_ip = f"http://{host_or_ip}"

            res = requests.post(f"{host_or_ip}/api/v1/vpn/register")
            res_json = res.json()

            if "vpn_auth_key" not in res_json:
                print("Registration failed", res)
                return {"status": "error"}

            status, error = connect_with_key(
                tailscale_host="http://tailscale:4000",
                headscale_host=f"{host_or_ip}/vpn",
                vpn_auth_key=res_json["vpn_auth_key"],
            )

            if status:
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
            status, vpn_auth_key = generate_key(headscale_host="http://headscale:4000")

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
    data = {"timeout": 5}
    command_url = f"{headscale_host}/commands/generate_key"
    try:
        resp = requests.post(command_url, json=data)
        report = get_result(json=resp.json())
        result_dict = dict(extract_nested_json(report))
        result = result_dict["Key"]

        # check if we got a key
        if len(result) == 48:
            return (True, result)
        else:
            return (False, "")

    except Exception as e:
        print("failed to make request", e)
        return (False, str(e))


def get_result(json: Dict) -> str:
    result_url = json.get("result_url", "")
    tries = 0
    limit = 5
    try:
        while True:
            print("Polling API Result", tries)
            result = requests.get(result_url)
            if "running" in result.text:
                time.sleep(1)
                tries += 1
                if tries > limit:
                    break
                continue
            else:
                return str(result.text)
    except Exception as e:
        print("Failed to get result", json, e)
    return ""
