# stdlib
import functools
import os
import subprocess
import sys
from typing import Any
from typing import Callable
from typing import Dict
from typing import Type

# third party
from flask import Flask
from flask import request
from flask_executor import Executor
from flask_executor.futures import Future
from flask_shell2http import Shell2HTTP

# Flask application instance
app = Flask(__name__)

executor = Executor(app)
shell2http = Shell2HTTP(app=app, executor=executor, base_url_prefix="/commands/")

hostname = os.environ.get("HOSTNAME", "node")  # default to node

# This is configuration for a "bridge" between the k8s network and the tailnet in both directions
# for outgoing packets, the source container needs to run with CAP_NET_ADMIN to add a route such as
#   ip route add 100.64.0.0/10 via 172.17.0.10
# in this example 172.17.0.10 is the IP address of the tailscale pod in this cluster
enable_proxy_to_traefik = os.environ.get("TAILSCALE_PROXY_TO_TRAEFIK", "disabled") == "enabled"
enable_gateway_to_tailnet = os.environ.get("TAILSCALE_GATEWAY_TO_TAILNET", "disabled") == "enabled"

key = os.environ.get("STACK_API_KEY", None)  # Get key from environment
if key is None:
    print("No STACK_API_KEY found, exiting.")
    sys.exit(1)


def check_stack_api_key(challenge_key: str) -> bool:
    key = os.environ.get("STACK_API_KEY", None)  # Get key from environment
    if key is None:
        return False
    if challenge_key == key:
        return True
    return False


def basic_auth_check(f: Any) -> Callable:
    @functools.wraps(f)
    def inner_decorator(*args: Any, **kwargs: Any) -> Type:
        stack_api_key = request.headers.get("X-STACK-API-KEY", "")
        if not check_stack_api_key(challenge_key=stack_api_key):
            raise Exception("STACK_API_KEY doesn't match.")
        return f(*args, **kwargs)

    return inner_decorator


def up_callback(context: Dict, future: Future) -> None:
    # optional user-defined callback function
    print(context, future.result())

    tailscale_ip = get_tailscale_ip()
    if enable_proxy_to_traefik:
        proxy_incoming(tailscale_ip, destination_ip=get_traefik_ip())
    if enable_gateway_to_tailnet:
        gateway_outgoing(tailscale_ip, tailscale_subnet="100.64.0.0/10")

def get_tailscale_ip() -> str:
    return subprocess.check_output(["tailscale", "ip", "-4"], text=True).strip()

def get_traefik_ip() -> str:
    return subprocess.check_output(
        "nslookup traefik.openmined.svc.cluster.local | grep ^Name -A1 | tail -1 | awk '{print $2}'",
        shell=True,
        text=True,
    ).strip()

def proxy_incoming(tailscale_ip: str, destination_ip: str) -> None:
    print(f"Adding iptables rule for DNAT {tailscale_ip} -> {destination_ip}")
    iptables_command = [
        "iptables",
        "-t", "nat",
        "-I", "PREROUTING",
        "-d", tailscale_ip,
        "-j", "DNAT",
        "--to-destination", destination_ip,
    ]
    print(subprocess.check_output(iptables_command, text=True))

def gateway_outgoing(tailscale_ip: str, tailscale_subnet: str) -> None:
    print(f"Adding iptables rule for SNAT {tailscale_ip} -> {tailscale_subnet}")
    iptables_command = [
        "iptables",
        "-t", "nat",
        "-A", "POSTROUTING",
        "-d", tailscale_subnet,
        "-o", "tailscale0",
        "-j", "SNAT",
        "--to-source", tailscale_ip,
    ]
    print(subprocess.check_output(iptables_command, text=True))

shell2http.register_command(
    endpoint="up",
    command_name="tailscale up",
    callback_fn=up_callback,
    decorators=[basic_auth_check],
)


def status_callback(context: Dict, future: Future) -> None:
    # optional user-defined callback function
    print(context, future.result())


shell2http.register_command(
    endpoint="status",
    command_name="tailscale status",
    callback_fn=status_callback,
    decorators=[basic_auth_check],
)
