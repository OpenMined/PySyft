# stdlib
import json
import re
import subprocess
import sys
import time
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

# third party
import requests


def get_container_port_mapping(container_name: str, port: int) -> str:
    cmd = "docker inspect --format='{{(index (index .NetworkSettings.Ports \""
    cmd += f"{port}/tcp"
    cmd += "\") 0).HostPort}}' "
    cmd += f"{container_name}"
    result = ""
    try:
        print(f"{container_name}:{port} -> ", end="")
        output = subprocess.check_output(cmd, shell=True)
        result = str(output.decode("utf-8")).strip()
        print(f"localhost:{result}")
    except Exception as e:
        print("Failed to get docker port mapping", e)
    return result


# the response contains a dict with a key called "report" which contains unescaped json
def extract_nested_json(nested_json: str) -> Union[Dict, List]:
    matcher = r'"report":"(.+)\\n"'
    parts = re.findall(matcher, nested_json)
    report_json = parts[0].replace("\\n", "").replace("\\t", "").replace('\\"', '"')
    return json.loads(report_json)


def generate_key(headscale_host: str) -> str:
    data = {"timeout": 5}
    result = ""
    command_url = f"{headscale_host}/commands/generate_key"
    try:
        resp = requests.post(command_url, json=data)
        report = get_result(json=resp.json())
        result_dict = dict(extract_nested_json(report))
        result = result_dict["key"]
    except Exception as e:
        print("failed to make request", e)
    return result


def connect_with_key(tailscale_host: str, headscale_host: str, authkey: str) -> str:
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
            f"{authkey}",
            "--accept-dns=false",
        ],
        "timeout": 60,
    }
    command_url = f"{tailscale_host}/commands/up"
    try:
        resp = requests.post(command_url, json=data)
        report = get_result(json=resp.json())
        return json.loads(report)["report"]
    except Exception as e:
        print("failed to make request", data, e)
    return ""


def list_nodes(headscale_host: str) -> List[Dict]:
    data = {"timeout": 5}
    result: List[Dict] = []
    command_url = f"{headscale_host}/commands/list_nodes"
    try:
        resp = requests.post(command_url, json=data)
        report = get_result(json=resp.json())
        result = list(extract_nested_json(report))
    except Exception as e:
        print("failed to make request", e)
    return result


def status(tailscale_host: str) -> str:
    data = {"timeout": 5}
    result = ""
    command_url = f"{tailscale_host}/commands/status"
    try:
        resp = requests.post(command_url, json=data)
        report = get_result(json=resp.json())
        result = json.loads(report)["report"]
    except Exception as e:
        print("failed to make request", e)
    return result


def get_result(json: Dict) -> str:
    result_url = json.get("result_url", "")
    tries = 0
    limit = 5
    try:
        while True:
            print("Polling API Result", tries)
            result = requests.get(result_url)
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
    return ""


# make sure you have a network node running:
# $ hagrid launch node network to docker:8081+
# then connect itself or another domain to the network
# $ python vpn/connect_vpn.py test_network_1_tailscale_1 test_network_1_headscale_1
# or if they are on different local docker networks
# $ python vpn/connect_vpn.py test_domain_1_tailscale_1 test_network_1_headscale_1 http://docker-host:8080
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Try: python vpn/connect_vpn.py domain_tailscale_1 network_headscale_1")
        sys.exit(1)

    tailscalehost = sys.argv[1]
    headscalehost = sys.argv[2]
    headscale_server: Optional[str] = None
    if len(sys.argv) > 3:
        headscale_server = sys.argv[3]

    if not tailscalehost.startswith("http") or not headscalehost.startswith("http"):
        # get the local mapped ports of the containers so we can make requests to them
        print("Finding local ports for containers")

    if not tailscalehost.startswith("http"):
        tailscale_api_port = get_container_port_mapping(
            container_name=tailscalehost, port=4000
        )
        tailscale_api = f"http://localhost:{tailscale_api_port}"
    else:
        tailscale_api = tailscalehost

    if not headscalehost.startswith("http"):
        headscale_api_port = get_container_port_mapping(
            container_name=headscalehost, port=4000
        )
        headscale_api = f"http://localhost:{headscale_api_port}"

        if headscale_server is None:
            headscale_port = get_container_port_mapping(
                container_name=headscalehost, port=8080
            )
            headscale_server = f"http://{headscalehost}:{headscale_port}"
    else:
        headscale_api = headscalehost

    if headscale_server is None:
        print("headscale Server required")
        sys.exit(1)

    print("\nConnecting with")
    print(f"tailscale API: {tailscale_api}")
    print(f"headscale API: {headscale_api}")
    print(f"headscale Server: {headscale_server}")

    authkey = generate_key(headscale_host=headscale_api)
    _ = connect_with_key(
        tailscale_host=tailscale_api, headscale_host=headscale_server, authkey=authkey
    )

    time.sleep(5)

    stat = status(tailscale_host=tailscale_api)

    for line in str(stat).split("\n"):
        matches = re.match(r"^\d.+", line)
        if matches is not None:
            stat_parts = re.split(r"(\s+)", matches.string)
            stat_ip = stat_parts[0]
            network_name = stat_parts[4]

            assert network_name == "omnet"
            break

    nodes = list_nodes(headscale_host=headscale_api)
    for node in nodes:
        ip_address = node["IPAddress"]
        hostname = node["HostInfo"]["Hostname"]

        if stat_ip == ip_address:
            print(f"Node: {hostname} Connected to {network_name} with IP: {ip_address}")
