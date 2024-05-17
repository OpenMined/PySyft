"""Python Level API to launch Syft services."""

# future
from __future__ import annotations

# stdlib
from collections.abc import Callable
from enum import Enum
import getpass
import inspect
import os
import sys
from typing import Any

# relative
from .abstract_node import NodeSideType
from .abstract_node import NodeType
from .client.client import login_as_guest as sy_login_as_guest
from .node.domain import Domain
from .node.enclave import Enclave
from .node.gateway import Gateway
from .node.server import serve_node
from .protocol.data_protocol import stage_protocol_changes
from .service.response import SyftError
from .util.util import get_random_available_port

DEFAULT_PORT = 8080
DEFAULT_URL = "http://localhost"

ClientAlias = Any  # we don't want to import Client in case it changes


def get_node_type(node_type: str | NodeType | None) -> NodeType | None:
    if node_type is None:
        node_type = os.environ.get("ORCHESTRA_NODE_TYPE", NodeType.DOMAIN)
    try:
        return NodeType(node_type)
    except ValueError:
        print(f"node_type: {node_type} is not a valid NodeType: {NodeType}")
    return None


def get_deployment_type(deployment_type: str | None) -> DeploymentType | None:
    if deployment_type is None:
        deployment_type = os.environ.get(
            "ORCHESTRA_DEPLOYMENT_TYPE", DeploymentType.PYTHON
        )

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
    REMOTE = "remote"


class NodeHandle:
    def __init__(
        self,
        node_type: NodeType,
        deployment_type: DeploymentType,
        node_side_type: NodeSideType,
        name: str,
        port: int | None = None,
        url: str | None = None,
        python_node: Any | None = None,
        shutdown: Callable | None = None,
    ) -> None:
        self.node_type = node_type
        self.name = name
        self.port = port
        self.url = url
        self.python_node = python_node
        self.shutdown = shutdown
        self.deployment_type = deployment_type
        self.node_side_type = node_side_type

    @property
    def client(self) -> Any:
        if self.port:
            return sy_login_as_guest(url=self.url, port=self.port)  # type: ignore
        elif self.deployment_type == DeploymentType.PYTHON:
            return self.python_node.get_guest_client(verbose=False)  # type: ignore
        else:
            raise NotImplementedError(
                f"client not implemented for the deployment type:{self.deployment_type}"
            )

    def login_as_guest(self, **kwargs: Any) -> ClientAlias:
        return self.client.login_as_guest(**kwargs)

    def login(
        self, email: str | None = None, password: str | None = None, **kwargs: Any
    ) -> ClientAlias:
        if not email:
            email = input("Email: ")

        if not password:
            password = getpass.getpass("Password: ")

        return self.client.login(email=email, password=password, **kwargs)

    def register(
        self,
        name: str,
        email: str | None = None,
        password: str | None = None,
        password_verify: str | None = None,
        institution: str | None = None,
        website: str | None = None,
    ) -> Any:
        if not email:
            email = input("Email: ")
        if not password:
            password = getpass.getpass("Password: ")
        if not password_verify:
            password_verify = getpass.getpass("Confirm Password: ")
        if password != password_verify:
            return SyftError(message="Passwords do not match")

        client = self.client
        return client.register(
            name=name,
            email=email,
            password=password,
            institution=institution,
            password_verify=password_verify,
            website=website,
        )

    def land(self) -> None:
        if self.deployment_type == DeploymentType.PYTHON:
            if self.shutdown:
                self.shutdown()
        else:
            print(
                f"Shutdown not implemented for the deployment type:{self.deployment_type}",
                file=sys.stderr,
            )


def deploy_to_python(
    node_type_enum: NodeType,
    deployment_type_enum: DeploymentType,
    port: int | str,
    name: str,
    host: str,
    reset: bool,
    tail: bool,
    dev_mode: bool,
    processes: int,
    local_db: bool,
    node_side_type: NodeSideType,
    enable_warnings: bool,
    n_consumers: int,
    thread_workers: bool,
    create_producer: bool = False,
    queue_port: int | None = None,
    association_request_auto_approval: bool = False,
    background_tasks: bool = False,
) -> NodeHandle:
    worker_classes = {
        NodeType.DOMAIN: Domain,
        NodeType.GATEWAY: Gateway,
        NodeType.ENCLAVE: Enclave,
    }

    if dev_mode:
        print("Staging Protocol Changes...")
        stage_protocol_changes()

    kwargs = {
        "name": name,
        "host": host,
        "port": port,
        "reset": reset,
        "processes": processes,
        "dev_mode": dev_mode,
        "tail": tail,
        "node_type": node_type_enum,
        "node_side_type": node_side_type,
        "enable_warnings": enable_warnings,
        "queue_port": queue_port,
        "n_consumers": n_consumers,
        "create_producer": create_producer,
        "association_request_auto_approval": association_request_auto_approval,
        "background_tasks": background_tasks,
    }

    if port:
        kwargs["in_memory_workers"] = True
        if port == "auto":
            port = get_random_available_port()
            kwargs["port"] = port

        sig = inspect.signature(serve_node)
        supported_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

        start, stop = serve_node(**supported_kwargs)
        start()
        return NodeHandle(
            node_type=node_type_enum,
            deployment_type=deployment_type_enum,
            name=name,
            port=port,
            url="http://localhost",
            shutdown=stop,
            node_side_type=node_side_type,
        )
    else:
        kwargs["local_db"] = local_db
        kwargs["thread_workers"] = thread_workers
        if node_type_enum in worker_classes:
            worker_class = worker_classes[node_type_enum]
            sig = inspect.signature(worker_class.named)
            supported_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            if "node_type" in sig.parameters.keys() and "migrate" in sig.parameters:
                supported_kwargs["migrate"] = True
            worker = worker_class.named(**supported_kwargs)
        else:
            raise NotImplementedError(f"node_type: {node_type_enum} is not supported")

        def stop() -> None:
            worker.stop()

        return NodeHandle(
            node_type=node_type_enum,
            deployment_type=deployment_type_enum,
            name=name,
            python_node=worker,
            node_side_type=node_side_type,
            shutdown=stop,
        )


def deploy_to_remote(
    node_type_enum: NodeType,
    deployment_type_enum: DeploymentType,
    name: str,
    node_side_type: NodeSideType,
) -> NodeHandle:
    node_port = int(os.environ.get("NODE_PORT", f"{DEFAULT_PORT}"))
    node_url = str(os.environ.get("NODE_URL", f"{DEFAULT_URL}"))
    return NodeHandle(
        node_type=node_type_enum,
        deployment_type=deployment_type_enum,
        name=name,
        port=node_port,
        url=node_url,
        node_side_type=node_side_type,
    )


class Orchestra:
    @staticmethod
    def launch(
        # node information and deployment
        name: str | None = None,
        node_type: str | NodeType | None = None,
        deploy_to: str | None = None,
        node_side_type: str | None = None,
        # worker related inputs
        port: int | str | None = None,
        processes: int = 1,  # temporary work around for jax in subprocess
        local_db: bool = False,
        dev_mode: bool = False,
        reset: bool = False,
        tail: bool = False,
        host: str | None = "0.0.0.0",  # nosec
        enable_warnings: bool = False,
        n_consumers: int = 0,
        thread_workers: bool = False,
        create_producer: bool = False,
        queue_port: int | None = None,
        association_request_auto_approval: bool = False,
        background_tasks: bool = False,
    ) -> NodeHandle:
        if dev_mode is True:
            thread_workers = True
        os.environ["DEV_MODE"] = str(dev_mode)

        node_type_enum: NodeType | None = get_node_type(node_type=node_type)
        node_side_type_enum = (
            NodeSideType.HIGH_SIDE
            if node_side_type is None
            else NodeSideType(node_side_type)
        )

        deployment_type_enum: DeploymentType | None = get_deployment_type(
            deployment_type=deploy_to
        )

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
                node_side_type=node_side_type_enum,
                enable_warnings=enable_warnings,
                n_consumers=n_consumers,
                thread_workers=thread_workers,
                create_producer=create_producer,
                queue_port=queue_port,
                association_request_auto_approval=association_request_auto_approval,
                background_tasks=background_tasks,
            )
        elif deployment_type_enum == DeploymentType.REMOTE:
            return deploy_to_remote(
                node_type_enum=node_type_enum,
                deployment_type_enum=deployment_type_enum,
                name=name,
                node_side_type=node_side_type_enum,
            )
        raise NotImplementedError(
            f"deployment_type: {deployment_type_enum} is not supported"
        )
