# stdlib
import re
import subprocess

# third party
import pytest


def docker_network_connect(direction: str = "connect") -> None:
    # this connects all the tailscale containers to the other docker compose project
    # networks thus allowing tailscale to find a direct route between them
    projects = [
        "network_1",
        "domain_1",
        "domain_2",
    ]
    for project in projects:
        for network in projects:
            try:
                if project == network:
                    continue
                container_name = f"test_{project}-tailscale-1"
                network_name = f"test_{network}_default"
                cmd = f"docker network {direction} {network_name} {container_name}"
                print(f"Connecting {container_name} to {network_name}")
                subprocess.call(cmd, shell=True)
            except Exception as e:
                print(f"Exception running: {cmd}. {e}")


@pytest.mark.security
def test_vpn_auth() -> None:
    # the tailscale container is currently the same so we can get away with a
    # single external scan
    containers = [
        "test_network_1-tailscale-1",
        # "test_domain_1-tailscale-1",
    ]
    stack_keys = ["stack_api_key"]

    # run in two containers so that all IPs are scanned externally
    for container in containers:
        try:
            container_hostname = f'echo hostname | docker exec -i {container} ash'
            cmd = f"curl -X GET https://{container_hostname} | jq .[].X-STACK-API-KEY | docker exec -i {container} ash"
            print(f"Scanning {container}")
            output = subprocess.check_output(cmd, shell=True)
            output = output.decode("utf-8")
        except Exception as e:
            print(f"Exception running: {cmd}. {e}")

        for key in stack_keys:
            matcher = re.compile("stack_api_key")
            lines = re.findall(matcher, output)
            assert len(lines) == 0
