# stdlib
import re
import subprocess


def docker_network_connect(cmd: str = "connect") -> None:
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
                cmd = f"docker network {cmd} {network_name} {container_name}"
                print(f"Connecting {container_name} to {network_name}")
                subprocess.call(cmd, shell=True)
            except Exception as e:
                print(f"Exception running: {cmd}. {e}")


def test_create_overlay_networks_docker() -> None:
    docker_network_connect(cmd="connect")


def test_vpn_scan() -> None:
    # we need to scan two containers so that the first container itself gets scanned
    # externally by another container to make sure all of the containers on the VPN
    # are externally scanned and their firewall rules verified
    containers = [
        "test_network_1-tailscale-1",
        "test_domain_1-tailscale-1",
    ]

    allowed_ports = [80]
    blocked_ports = [21, 4000, 8001, 8011, 8080, 5050, 5432, 5555, 5672, 15672]

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


def test_remove_overlay_networks_docker() -> None:
    docker_network_connect(cmd="disconnect")
