"""Python Level API to launch Docker Containers using Hagrid"""
# future
from __future__ import annotations

# stdlib
from enum import Enum
import getpass
import os
import subprocess  # nosec
from typing import Any
from typing import Callable
from typing import Optional
from typing import Union

# third party
import gevent

# relative
from .cli import str_to_bool
from .grammar import find_available_port
from .names import random_name
from .util import shell

try:
    # syft absolute
    from syft.abstract_node import NodeType
except Exception:
    print("Please install syft with `pip install syft`")
    pass

DEFAULT_PORT = 8080
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


def container_exists(name: str) -> bool:
    output = shell(f"docker ps -q -f name='{name}'")
    return len(output) > 0


def container_id(name: str) -> Optional[str]:
    output = shell(f"docker ps -q -f name='{name}'")
    if len(output) > 0:
        return output[0].strip()
    return None


def port_from_container(name: str) -> Optional[int]:
    cid = container_id(name)
    if cid is None:
        return None
    output = shell(f"docker port {cid}")
    if len(output) > 0:
        try:
            # 80/tcp -> 0.0.0.0:8080
            lines = output.split("\n")
            parts = lines[0].split(":")
            port = int(parts[1].strip())
            return port
        except Exception:  # nosec
            return None
    return None


def container_exists_with(name: str, port: int) -> bool:
    output = shell(f"docker ps -q -f name='{name}' -f expose='{port}'")
    return len(output) > 0


def get_node_type(node_type: Optional[str]) -> Optional[NodeType]:
    if node_type is None:
        node_type = os.environ.get("ORCHESTRA_NODE_TYPE", NodeType.DOMAIN)
    try:
        return NodeType(node_type)
    except ValueError:
        print(f"node_type: {node_type} is not a valid NodeType: {NodeType}")
    return None


def get_deployment_type(deployment_type: Optional[str]) -> Optional[DeploymentType]:
    if deployment_type is None:
        deployment_type = os.environ.get(
            "ORCHESTRA_DEPLOYMENT_TYPE", DeploymentType.PYTHON
        )

    # provide shorthands
    if deployment_type == "container":
        deployment_type = "container_stack"

    try:
        return DeploymentType(deployment_type)
    except ValueError:
        print(
            f"deployment_type: {deployment_type} is not a valid DeploymentType: {DeploymentType}"
        )
    return None


# Can also be specified by the environment variable
# ORCHESTRA_DEPLOYMENT_TYPE
class DeploymentType(Enum):
    PYTHON = "python"
    SINGLE_CONTAINER = "single_container"
    CONTAINER_STACK = "container_stack"
    K8S = "k8s"


class NodeHandle:
    def __init__(
        self,
        node_type: NodeType,
        deployment_type: DeploymentType,
        name: str,
        port: Optional[int] = None,
        url: Optional[str] = None,
        python_node: Optional[Any] = None,
        shutdown: Optional[Callable] = None,
    ) -> None:
        self.node_type = node_type
        self.name = name
        self.port = port
        self.url = url
        self.python_node = python_node
        self.shutdown = shutdown
        self.deployment_type = deployment_type

    @property
    def client(self) -> Any:
        if self.port:
            sy = get_syft_client()
            return sy.login(url=self.url, port=self.port, verbose=False)  # type: ignore
        elif self.deployment_type == DeploymentType.PYTHON:
            return self.python_node.get_guest_client(verbose=False)  # type: ignore

    def login(
        self, email: Optional[str] = None, password: Optional[str] = None
    ) -> Optional[Any]:
        client = self.client
        if email and password:
            return client.login(email=email, password=password)
        return None

    def register(
        self,
        name: str,
        email: str,
        password: str,
        institution: Optional[str] = None,
        website: Optional[str] = None,
    ) -> Any:
        client = self.client
        return client.register(
            name=name,
            email=email,
            password=password,
            institution=institution,
            website=website,
        )

    def land(self) -> None:
        if self.deployment_type == DeploymentType.PYTHON:
            if self.shutdown:
                self.shutdown()
        elif self.deployment_type == DeploymentType.VM:
            pass
        else:
            Orchestra.land(self.name, deployment_type=self.deployment_type)


def deploy_to_python(
    node_type_enum: NodeType,
    deployment_type_enum: DeploymentType,
    port: Union[int, str],
    name: str,
    host: str,
    reset: bool,
    tail: bool,
    dev_mode: bool,
    processes: int,
    local_db: bool,
) -> NodeHandle:
    sy = get_syft_client()
    if port:
        if port == "auto":
            # dont use default port to prevent port clashes in CI
            port = find_available_port(host="localhost", port=None, search=True)
        start, stop = sy.serve_node(  # type: ignore
            name=name,
            host=host,
            port=port,
            reset=reset,
            dev_mode=dev_mode,
            tail=tail,
            node_type=node_type_enum,
        )
        start()
        return NodeHandle(
            node_type=node_type_enum,
            deployment_type=deployment_type_enum,
            name=name,
            port=port,
            url="http://localhost",
            shutdown=stop,
        )
    else:
        worker = sy.Worker.named(  # type: ignore
            name=name,
            processes=processes,
            reset=reset,
            local_db=local_db,
            node_type=node_type_enum,
        )
        return NodeHandle(
            node_type=node_type_enum,
            deployment_type=deployment_type_enum,
            name=name,
            python_node=worker,
        )


def deploy_to_vm(
    node_type_enum: NodeType, deployment_type_enum: DeploymentType, name: str
) -> NodeHandle:
    # Q: Ask Madhava, on why we have a fixed IP address for VM.
    return NodeHandle(
        node_type=node_type_enum,
        deployment_type=deployment_type_enum,
        name=name,
        port=80,
        url="http://192.168.56.2",
    )


def deploy_to_k8s(
    node_type_enum: NodeType, deployment_type_enum: DeploymentType, name: str
) -> NodeHandle:
    node_port = int(os.environ.get("NODE_PORT", f"{DEFAULT_PORT}"))
    return NodeHandle(
        node_type=node_type_enum,
        deployment_type=deployment_type_enum,
        name=name,
        port=node_port,
        url="http://localhost",
    )


def deploy_to_container(
    node_type_enum: NodeType,
    deployment_type_enum: DeploymentType,
    reset: bool,
    cmd: bool,
    tail: bool,
    verbose: bool,
    tag: str,
    render: bool,
    dev_mode: bool,
    port: Union[int, str],
    name: str,
) -> Optional[NodeHandle]:
    if port == "auto" or port is None:
        if container_exists(name=name):
            port = port_from_container(name=name)  # type: ignore
        else:
            port = find_available_port(host="localhost", port=DEFAULT_PORT, search=True)

    # Currently by default we launch in dev mode
    if reset:
        Orchestra.reset(name, deployment_type_enum)
    else:
        if container_exists_with(name=name, port=port):
            return NodeHandle(
                node_type=node_type_enum,
                deployment_type=deployment_type_enum,
                name=name,
                port=port,
                url="http://localhost",
            )

    # Start a subprocess and capture its output
    commands = ["hagrid", "launch"]

    name = random_name() if not name else name
    commands.extend([name, node_type_enum.value])

    commands.append("to")
    commands.append(f"docker:{port}")

    if dev_mode:
        commands.append("--dev")

    # by default , we deploy as container stack
    if deployment_type_enum == DeploymentType.SINGLE_CONTAINER:
        commands.append("--deploy=single_container")

    if cmd:
        commands.append("--cmd")

    if tail:
        commands.append("--tail")

    if verbose:
        commands.append("--verbose")

    if tag:
        commands.append(f"--tag={tag}")

    if render:
        commands.append("--render")

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
    gevent.joinall([stdout_thread, stderr_thread], raise_error=True)

    if not cmd:
        return NodeHandle(
            node_type=node_type_enum,
            deployment_type=deployment_type_enum,
            name=name,
            port=port,
            url="http://localhost",
        )
    return None


class Orchestra:
    @staticmethod
    def launch(
        # node information and deployment
        name: Optional[str] = None,
        node_type: Optional[str] = None,
        deploy_to: Optional[str] = None,
        # worker related inputs
        port: Optional[Union[int, str]] = None,
        processes: int = 1,  # temporary work around for jax in subprocess
        local_db: bool = False,
        dev_mode: bool = False,
        cmd: bool = False,
        reset: bool = False,
        tail: bool = False,
        host: Optional[str] = "0.0.0.0",  # nosec
        tag: Optional[str] = "latest",
        verbose: bool = False,
        render: bool = False,
    ) -> Optional[NodeHandle]:
        if dev_mode is True:
            os.environ["DEV_MODE"] = "True"

        dev_mode = str_to_bool(os.environ.get("DEV_MODE", f"{dev_mode}"))

        node_type_enum: Optional[NodeType] = get_node_type(node_type=node_type)
        if not node_type_enum:
            return None

        deployment_type_enum: Optional[DeploymentType] = get_deployment_type(
            deployment_type=deploy_to
        )
        if not deployment_type_enum:
            return None

        if deployment_type_enum == DeploymentType.PYTHON:
            return deploy_to_python(
                node_type_enum=node_type_enum,
                deployment_type_enum=deployment_type_enum,
                port=port,
                name=name,
                host=host,
                reset=reset,
                tail=tail,
                dev_mode=dev_mode,
                processes=processes,
                local_db=local_db,
            )

        elif deployment_type_enum == DeploymentType.VM:
            return deploy_to_vm(
                node_type_enum=node_type_enum,
                deployment_type_enum=deployment_type_enum,
                name=name,
            )

        elif deployment_type_enum == DeploymentType.K8S:
            return deploy_to_k8s(
                node_type_enum=node_type_enum,
                deployment_type_enum=deployment_type_enum,
                name=name,
            )

        elif (
            deployment_type_enum == DeploymentType.CONTAINER_STACK
            or deployment_type_enum == DeploymentType.SINGLE_CONTAINER
        ):
            return deploy_to_container(
                node_type_enum=node_type_enum,
                deployment_type_enum=deployment_type_enum,
                reset=reset,
                cmd=cmd,
                tail=tail,
                verbose=verbose,
                tag=tag,
                render=render,
                dev_mode=dev_mode,
                port=port,
                name=name,
            )
        else:
            print(f"deployment_type: {deployment_type_enum} is not supported")
            return None

    @staticmethod
    def land(
        name: str, deployment_type: Union[str, DeploymentType], reset: bool = False
    ) -> None:
        deployment_type_enum = DeploymentType(deployment_type)
        Orchestra.shutdown(name=name, deployment_type_enum=deployment_type_enum)
        if reset:
            Orchestra.reset(name, deployment_type_enum=deployment_type_enum)

    @staticmethod
    def shutdown(name: str, deployment_type_enum: DeploymentType) -> None:
        if deployment_type_enum != DeploymentType.PYTHON:
            snake_name = to_snake_case(name)

            land_output = shell(f"hagrid land {snake_name} --force")
            if "Removed" in land_output:
                print(f" ✅ {snake_name} Container Removed")
            else:
                print(f"❌ Unable to remove container: {snake_name} :{land_output}")

    @staticmethod
    def reset(name: str, deployment_type_enum: DeploymentType) -> None:
        if deployment_type_enum == DeploymentType.PYTHON:
            sy = get_syft_client()
            _ = sy.Worker.named(name, processes=1, reset=True)  # type: ignore
        elif (
            deployment_type_enum == DeploymentType.CONTAINER_STACK
            or deployment_type_enum == DeploymentType.SINGLE_CONTAINER
        ):
            if container_exists(name=name):
                Orchestra.shutdown(name=name, deployment_type_enum=deployment_type_enum)

            snake_name = to_snake_case(name)

            volume_output = shell(
                f"docker volume rm {snake_name}_credentials-data --force || true"
            )

            if "Error" not in volume_output:
                print(f" ✅ {snake_name} Volume Removed")
            else:
                print(
                    f"❌ Unable to remove container volume: {snake_name} :{volume_output}"
                )
        else:
            raise NotImplementedError(
                f"Reset not implemented for the deployment type:{deployment_type_enum}"
            )
