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
                container_name = f"test_{project}_tailscale_1"
                network_name = f"test_{network}_default"
                cmd = f"docker network {direction} {network_name} {container_name}"
                print(f"Connecting {container_name} to {network_name}")
                subprocess.call(cmd, shell=True)
            except Exception as e:
                print(f"Exception running: {cmd}. {e}")


@pytest.mark.security
def test_create_overlay_networks_docker() -> None:
    docker_network_connect(direction="connect")


@pytest.mark.security
def test_vpn_scan() -> None:
    # the tailscale container is currently the same so we can get away with a
    # single external scan
    containers = [
        "test_network_1_tailscale_1",
        # "test_domain_1-tailscale-1",
    ]

    allowed_ports = [80]
    blocked_ports = [21, 4000, 8001, 8011, 8080, 5050, 5432, 5555, 5672, 15672]
    # when SSL is enabled we route 80 to 81 externally so that we can redirect to HTTPS
    # however we want to leave normal port 80 available over the VPN internally
    blocked_ports.append(81)  # this shouldnt be available over the VPN
    # SSL shouldnt be available over the VPN since IPs cant be used in certs
    # in this case we might be bound to any number of ports during dev mode from 443+
    for i in range(443, 451):
        blocked_ports.append(i)

    # run in two containers so that all IPs are scanned externally
    for container in containers:
        try:
            cmd = f"cat scripts/vpn_scan.sh | docker exec -i {container} ash"
            print(f"Scanning {container}")
            output = subprocess.check_output(cmd, shell=True)
            output = output.decode("utf-8")
        except Exception as e:
            print(f"Exception running: {cmd}. {e}")

        for port in allowed_ports:
            matcher = re.compile(f" {port} port .+succeeded")
            lines = re.findall(matcher, output)
            assert len(lines) > 0

        for port in blocked_ports:
            matcher = re.compile(f" {port} port .+succeeded")
            lines = re.findall(matcher, output)
            assert len(lines) == 0


@pytest.mark.security
def test_remove_overlay_networks_docker() -> None:
    docker_network_connect(direction="disconnect")
