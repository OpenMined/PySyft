# stdlib
import ipaddress
import os
import platform
import subprocess
import time

# third party
import pytest

# syft absolute
import syft as sy

CONTAINER_HOST = os.environ.get("CONTAINER_HOST", "docker")
EMULATION = os.environ.get("EMULATION", "false")
OS = platform.system().lower()
print("CONTAINER_HOST", CONTAINER_HOST)
print("EMULATION", EMULATION)


def run_command(container: str, cmd: str) -> str:
    try:
        docker_cmd = f"docker exec -i {container} ash -c '{cmd}'"
        output = subprocess.check_output(docker_cmd, shell=True)
        output = output.decode("utf-8").strip()
        return output
    except Exception as e:
        print(f"Exception running: {docker_cmd}. {e}")
        raise e


def get_ip_of_container(container: str) -> str:
    print(f"Getting IP for {container}")

    get_ip_cmd = (
        r"""ip addr show eth0 | grep "inet\b" | awk '\''{print $2}'\'' | cut -d/ -f1"""
    )
    output = run_command(container, get_ip_cmd)
    parts = output.split(" ")
    output = parts[0]
    ip_addr = ipaddress.ip_address(output)
    print(f"Got {ip_addr}")
    return output


def curl_ip(container: str, ip_addr: str) -> str:
    install_curl_cmd = """
    apk -U upgrade || true
    apk fix || true
    apk add curl
    """

    print(f"Installing CURL into {container}")
    output = run_command(container, install_curl_cmd)
    print(output)

    run_curl_cmd = f"curl -v {ip_addr} -o /dev/null -w "
    run_curl_cmd += r"'%{http_code}' -s"
    print(f"Running CURL from {container} to {ip_addr}")
    output = run_command(container, run_curl_cmd)
    print(output)

    try:
        return int(output)
    except Exception as e:
        print(f"Failed to convert HTTP STATUS: {output} to int")
        raise e


def print_firewall_rules():
    blocked, allowed = sy.client.client.SyftClient.get_ip_rules()
    print(f"Blocked IPs: {[str(ip) for ip in blocked]}")
    print(f"Allowed IPs: {[str(ip) for ip in allowed]}")


@pytest.mark.network
def test_firewall() -> None:
    if OS == "windows" or CONTAINER_HOST != "docker" or EMULATION != "false":
        return

    proxy_container = "test_domain_1-proxy-1"
    frontend_container = "test_domain_1-frontend-1"

    proxy_ip = get_ip_of_container(proxy_container)
    frontend_ip = get_ip_of_container(frontend_container)

    print("Firewall rules before test")
    print_firewall_rules()
    # access allowed
    print(f"Unblocking IP {frontend_ip}")
    sy.client.client.SyftClient.unblock_ip(ip=ipaddress.ip_address(frontend_ip))
    print_firewall_rules()
    time.sleep(2)
    result = curl_ip(frontend_container, proxy_ip)
    assert result == 200, f"{frontend_ip} is blocked"

    # access denied
    print(f"Blocking IP {frontend_ip}")
    sy.client.client.SyftClient.block_ip(ip=ipaddress.ip_address(frontend_ip))
    print_firewall_rules()
    time.sleep(2)
    result = curl_ip(frontend_container, proxy_ip)
    assert result == 403, f"{frontend_ip} is allowed"

    # access allowed again
    print(f"Unblocking IP {frontend_ip}")

    sy.client.client.SyftClient.unblock_ip(ip=ipaddress.ip_address(frontend_ip))
    print_firewall_rules()
    time.sleep(2)
    result = curl_ip(frontend_container, proxy_ip)
    assert result == 200, f"{frontend_ip} is blocked"
