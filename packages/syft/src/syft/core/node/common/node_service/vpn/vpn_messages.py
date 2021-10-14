# future
from __future__ import annotations

# stdlib
import json
import time
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

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
                authkey=vpn_auth_key,
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
    tailscale_host: str, headscale_host: str, authkey: str
) -> Tuple[bool, str]:
    data = {
        "args": [
            "-login-server",
            f"{headscale_host}",
            "--reset",
            "--force-reauth",
            "--authkey",
            f"{authkey}",
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


def get_result(json: Dict) -> str:
    result_url = json.get("result_url")
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
