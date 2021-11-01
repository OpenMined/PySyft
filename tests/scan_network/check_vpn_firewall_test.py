# stdlib
import re
import subprocess


def test_create_overlay_networks_docker() -> None:
    # this connects all the tailscale containers to the other docker compose project
    # networks thus allowing tailscale to find a direct route between them
    containers = [
        "test_network_1-tailscale-1",
        "test_domain_1-tailscale-1",
        "test_domain_2-tailscale-1",
    ]
    networks = [
        "test_network_1_default",
        "test_domain_1_default",
        "test_domain_2_default",
    ]
    for container in containers:
        for network in networks:
            try:
                cmd = f"docker network connect {network} {container}"
                print(f"Connecting {container} to {network}")
                subprocess.call(cmd, shell=True)
            except Exception as e:
                print(f"Exception running: {cmd}. {e}")


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
