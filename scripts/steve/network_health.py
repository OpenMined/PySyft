# stdlib
from enum import Enum
import json

# third party
import paramiko
import requests

# syft absolute
import syft as sy
from syft.grid.grid_url import GridURL


class EndPoints(Enum):
    METADATA = "api/v1/syft/metadata"
    STATUS = "api/v1/status"


class Status(Enum):
    OK = "Ok"
    FAILED = "Failed"


PUBLIC_NETWORKS_URL = (
    "https://raw.githubusercontent.com/OpenMined/NetworkRegistry/main/networks.json"
)


def get_listed_public_networks(url=PUBLIC_NETWORKS_URL):
    response = requests.get(url, timeout=5)
    print(f"GET Public Network Status: {response.status_code}")
    response = response.json()
    network_list = response.get("networks", [])[1:]
    return network_list


def check_network_status(host_url):
    url = f"http://{host_url}/{EndPoints.STATUS.value}"
    response = requests.get(url, timeout=0.5)
    if response.status_code == 200:
        return Status.OK.value
    return Status.FAILED.value


def check_metadata_api(host_url):
    url = f"http://{host_url}/{EndPoints.METADATA.value}"
    response = requests.get(url, timeout=5)
    if response.status_code == 200:
        return Status.OK.value

    return Status.FAILED.value


def check_login_via_syft(host_url, retry=5):
    network_client = None
    grid_url = GridURL(host_or_ip=host_url)
    grid_url = grid_url.with_path("/api/v1")
    while not network_client and retry > 0:

        try:
            network_client = sy.connect(url=grid_url, timeout=5)
        except TimeoutError:
            retry -= 1
    if network_client is None:
        return Status.FAILED.value

    print(network_client.name)
    return Status.OK.value


def check_ip_port(host_ip: str, port: int) -> bool:
    # stdlib
    import socket

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host_ip, port))
        sock.close()
        if result == 0:
            return Status.OK.value
    except Exception:
        pass
    return Status.FAILED.value


def check_tailscale_status(host_ip: str, key_filename: str):
    ssh_client = paramiko.SSHClient()

    # To avoid an "unknown hosts" error. Solve this differently if you must...
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # This mechanism uses a key_filename.
    username = "azureuser"

    ssh_client.connect(hostname=host_ip, username=username, key_filename=key_filename)

    command = "TAILSCALE_CONTAINER_NAME=$(sudo docker ps --format '{{.Names}}' | grep -m 1 tailscale) &&"
    command += "sudo docker exec $TAILSCALE_CONTAINER_NAME tailscale status"

    stdin, stdout, stderr = ssh_client.exec_command(command)

    errors = stderr.readlines()

    if len(errors) > 0:
        return Status.FAILED.value

    stdout = stdout.readlines()
    if len(stdout) > 0:
        return Status.OK.value

    return Status.FAILED.value


def check_headscale_node_list(host_ip: str, key_filename: str):
    ssh_client = paramiko.SSHClient()

    # To avoid an "unknown hosts" error. Solve this differently if you must...
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # This mechanism uses a key_filename.
    username = "azureuser"

    ssh_client.connect(hostname=host_ip, username=username, key_filename=key_filename)

    command = "HEADSCALE_CONTAINER_NAME=$(sudo docker ps --format '{{.Names}}' | grep -m 1 headscale) && "
    command += "sudo docker exec $HEADSCALE_CONTAINER_NAME headscale nodes list -o json"

    _, stdout, stderr = ssh_client.exec_command(command)

    errors = stderr.readlines()
    print(errors)

    if len(errors) > 0:
        return Status.FAILED.value

    stdout = json.loads(stdout.read())
    if len(stdout) > 0:
        return stdout
    return Status.FAILED.value
