# stdlib
import json
import os
import re
import time
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
import requests

# relative
from ......logger import critical


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
    tailscale_host: Optional[str] = "http://tailscale:4000",
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


def connect_with_key(
    headscale_host: str,
    vpn_auth_key: str,
    tailscale_host: Optional[str] = "http://tailscale:4000",
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


def disconnect(
    tailscale_host: Optional[str] = "http://tailscale:4000",
) -> Tuple[bool, str]:
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


def generate_key(
    headscale_host: Optional[str] = "http://headscale:4000",
) -> Tuple[bool, str]:
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


# the response contains a dict with a key called "report" which contains unescaped json
def extract_nested_json(nested_json: str) -> Union[Dict, List]:
    matcher = r'"report":"(.+)\\n"'
    parts = re.findall(matcher, nested_json)
    report_json = parts[0].replace("\\n", "").replace("\\t", "").replace('\\"', '"')
    return json.loads(report_json)
