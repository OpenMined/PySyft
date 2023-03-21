"""Python Level API to launch Docker Containers using Hagrid"""
# future
from __future__ import annotations

# stdlib
from enum import Enum
import getpass
import os
import subprocess  # nosec
from typing import Any
from typing import Optional

# third party
import gevent

# relative
from .grammar import find_available_port
from .names import random_name
from .util import shell

# Gevent used instead of threading module ,as we monkey patch gevent in syft
# and this causes context switch error when we use normal threading in hagrid


# Define a function to read and print a stream
def read_stream(stream: subprocess.PIPE) -> None:
    while True:
        line = stream.readline()
        if not line:
            break
        print(line, end="")
        gevent.sleep(0)


def to_snake_case(name: str) -> str:
    return name.lower().replace(" ", "_")


def get_syft_client() -> Optional[Any]:
    try:
        # syft absolute
        import syft as sy

        return sy
    except Exception:
        print("Please install syft with `pip install syft`")
        pass
    return None


class NodeType(Enum):
    GATEWAY = "gateway"
    DOMAIN = "domain"
    WORKER = "worker"
    ENCLAVE = "enclave"
    PYTHON = "python"


class NodeHandle:
    def __init__(
        self,
        node_type: NodeType,
        name: str,
        port: Optional[int] = None,
        url: Optional[str] = None,
        python_node: Optional[Any] = None,
    ) -> None:
        self.node_type = node_type
        self.name = name
        self.port = port
        self.url = url
        self.python_node = python_node

    @property
    def client(self) -> Any:
        if self.node_type == NodeType.PYTHON:
            return self.python_node.guest_client  # type: ignore
        else:
            sy = get_syft_client()
            return sy.login(port=self.port)  # type: ignore

    def login(
        self, email: Optional[str] = None, password: Optional[str] = None
    ) -> Optional[Any]:
        client = self.client
        if email and password:
            return client.login(email=email, password=password)
        return None


def get_node_type(node_type: Optional[str]) -> Optional[NodeType]:
    if node_type is None:
        node_type = os.environ.get("ORCHESTRA_NODE_TYPE", NodeType.PYTHON)
    try:
        return NodeType(node_type)
    except ValueError:
        print(f"node_type: {node_type} is not a valid NodeType: {NodeType}")
    return None


class Orchestra:
    @staticmethod
    def launch(
        name: Optional[str] = None,
        node_type: Optional[str] = None,
        dev_mode: bool = True,
        cmd: bool = False,
        reset: bool = False,
        tail: bool = False,
        port: Optional[int] = 8080,
        processes: int = 1,  # temporary work around for jax in subprocess
    ) -> Optional[NodeHandle]:
        node_type_enum: Optional[NodeType] = get_node_type(node_type=node_type)
        if not node_type_enum:
            return None

        if node_type_enum == NodeType.PYTHON:
            sy = get_syft_client()
            worker = sy.Worker.named(name, processes=processes, reset=reset)  # type: ignore
            return NodeHandle(node_type=node_type_enum, name=name, python_node=worker)

        # Currently by default we launch in dev mode
        if reset:
            Orchestra.reset(name)

        # Start a subprocess and capture its output
        commands = ["hagrid", "launch"]

        name = random_name() if not name else name
        commands.extend([name, node_type_enum.value])

        if port is None:
            port = find_available_port(host="localhost", port=port, search=True)

        commands.append("to")
        commands.append(f"docker:{port}")

        if dev_mode:
            commands.append("--dev")

        if cmd:
            commands.append("--cmd")

        if tail:
            commands.append("--tail")

        # needed for building containers
        USER = os.environ.get("USER", getpass.getuser())
        env = os.environ.copy()
        env["USER"] = USER

        process = subprocess.Popen(  # nosec
            commands, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
        )
        # Start gevent threads to read and print the output and error streams
        stdout_thread = gevent.spawn(read_stream, process.stdout)
        stderr_thread = gevent.spawn(read_stream, process.stderr)

        # Wait for the threads to finish
        gevent.joinall([stdout_thread, stderr_thread])

        if not cmd:
            return NodeHandle(
                node_type=node_type_enum, name=name, port=port, url="http://localhost"
            )
        return None

    @staticmethod
    def land(name: str, node_type: Optional[str] = None) -> None:
        Orchestra.reset(name, node_type=node_type)

    @staticmethod
    def reset(name: str, node_type: Optional[str] = None) -> None:
        node_type_enum: Optional[NodeType] = get_node_type(node_type=node_type)
        if node_type_enum == NodeType.PYTHON:
            sy = get_syft_client()
            _ = sy.Worker.named(name, processes=1, reset=True)  # type: ignore
        else:
            snake_name = to_snake_case(name)

            land_output = shell(f"hagrid land {snake_name} --force")
            if "Removed" in land_output:
                print(f" ✅ {snake_name} Container Removed")
            else:
                print(f"❌ Unable to remove container: {snake_name} :{land_output}")

            volume_output = shell(
                f"docker volume rm {snake_name}_credentials-data --force || true"
            )

            if "Error" not in volume_output:
                print(f" ✅ {snake_name} Volume Removed")
            else:
                print(
                    f"❌ Unable to remove container volume: {snake_name} :{volume_output}"
                )
