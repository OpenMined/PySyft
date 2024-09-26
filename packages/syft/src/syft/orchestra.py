"""Python Level API to launch Syft services."""

# future
from __future__ import annotations

# stdlib
from collections.abc import Callable
import getpass
import inspect
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

# third party
from IPython.display import display

# relative
from .abstract_server import ServerSideType
from .abstract_server import ServerType
from .client.client import login as sy_login
from .client.client import login_as_guest as sy_login_as_guest
from .deployment_type import DeploymentType
from .protocol.data_protocol import stage_protocol_changes
from .server.datasite import Datasite
from .server.enclave import Enclave
from .server.gateway import Gateway
from .server.uvicorn import serve_server
from .service.queue.queue import ConsumerType
from .service.response import SyftInfo
from .types.errors import SyftException
from .util.util import get_random_available_port

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8080
DEFAULT_URL = "http://localhost"

ClientAlias = Any  # we don't want to import Client in case it changes


def get_server_type(server_type: str | ServerType | None) -> ServerType | None:
    if server_type is None:
        server_type = os.environ.get("ORCHESTRA_SERVER_TYPE", ServerType.DATASITE)
    try:
        return ServerType(server_type)
    except ValueError:
        print(f"server_type: {server_type} is not a valid ServerType: {ServerType}")
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


class ServerHandle:
    def __init__(
        self,
        server_type: ServerType,
        deployment_type: DeploymentType,
        server_side_type: ServerSideType,
        name: str,
        port: int | None = None,
        url: str | None = None,
        python_server: Any | None = None,
        shutdown: Callable | None = None,
    ) -> None:
        self.server_type = server_type
        self.name = name
        self.port = port
        self.url = url
        self.python_server = python_server
        self.shutdown = shutdown
        self.deployment_type = deployment_type
        self.server_side_type = server_side_type

    @property
    def client(self) -> Any:
        if self.port:
            return sy_login_as_guest(url=self.url, port=self.port)  # type: ignore
        elif self.deployment_type == DeploymentType.PYTHON:
            return self.python_server.get_guest_client(verbose=False)  # type: ignore
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

        if self.port:
            return sy_login(
                email=email, password=password, url=self.url, port=self.port
            )  # type: ignore
        elif self.deployment_type == DeploymentType.PYTHON:
            guest_client = self.python_server.get_guest_client(verbose=False)  # type: ignore
            return guest_client.login(email=email, password=password, **kwargs)  # type: ignore
        else:
            raise NotImplementedError(
                f"client not implemented for the deployment type:{self.deployment_type}"
            )

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
            raise SyftException(public_message="Passwords do not match")

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
    server_type_enum: ServerType,
    deployment_type_enum: DeploymentType,
    port: int | str,
    name: str,
    host: str,
    reset: bool,
    tail: bool,
    dev_mode: bool,
    processes: int,
    server_side_type: ServerSideType,
    enable_warnings: bool,
    n_consumers: int,
    thread_workers: bool,
    create_producer: bool = False,
    queue_port: int | None = None,
    association_request_auto_approval: bool = False,
    background_tasks: bool = False,
    log_level: str | int | None = None,
    debug: bool = False,
    migrate: bool = False,
    consumer_type: ConsumerType | None = None,
    db_url: str | None = None,
) -> ServerHandle:
    worker_classes = {
        ServerType.DATASITE: Datasite,
        ServerType.GATEWAY: Gateway,
        ServerType.ENCLAVE: Enclave,
    }

    if dev_mode:
        logger.debug("Staging Protocol Changes...")
        stage_protocol_changes()

    kwargs = {
        "name": name,
        "host": host,
        "port": port,
        "reset": reset,
        "processes": processes,
        "dev_mode": dev_mode,
        "tail": tail,
        "server_type": server_type_enum,
        "server_side_type": server_side_type,
        "enable_warnings": enable_warnings,
        "queue_port": queue_port,
        "n_consumers": n_consumers,
        "create_producer": create_producer,
        "association_request_auto_approval": association_request_auto_approval,
        "log_level": log_level,
        "background_tasks": background_tasks,
        "debug": debug,
        "migrate": migrate,
        "deployment_type": deployment_type_enum,
        "consumer_type": consumer_type,
        "db_url": db_url,
    }

    if port:
        kwargs["in_memory_workers"] = True
        if port == "auto":
            port = get_random_available_port()
        else:
            try:
                port = int(port)
            except ValueError:
                raise ValueError(
                    f"port must be either 'auto' or a valid int not: {port}"
                )
        kwargs["port"] = port

        sig = inspect.signature(serve_server)
        supported_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

        start, stop = serve_server(**supported_kwargs)
        start()
        return ServerHandle(
            server_type=server_type_enum,
            deployment_type=deployment_type_enum,
            name=name,
            port=port,
            url="http://localhost",
            shutdown=stop,
            server_side_type=server_side_type,
        )
    else:
        kwargs["thread_workers"] = thread_workers
        if server_type_enum in worker_classes:
            worker_class = worker_classes[server_type_enum]
            sig = inspect.signature(worker_class.named)
            supported_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            if "server_type" in sig.parameters.keys() and "migrate" in sig.parameters:
                supported_kwargs["migrate"] = migrate
            worker = worker_class.named(**supported_kwargs)
        else:
            raise NotImplementedError(
                f"server_type: {server_type_enum} is not supported"
            )

        def stop() -> None:
            worker.stop()

        return ServerHandle(
            server_type=server_type_enum,
            deployment_type=deployment_type_enum,
            name=name,
            python_server=worker,
            server_side_type=server_side_type,
            shutdown=stop,
        )


def deploy_to_remote(
    server_type_enum: ServerType,
    deployment_type_enum: DeploymentType,
    name: str,
    server_side_type: ServerSideType,
    host: str | None = None,
    port: int | None = None,
    migrate: bool = False,
) -> ServerHandle:
    if migrate:
        raise ValueError("Cannot migrate via orchestra on remote server")

    # Preference order: Environment Variable > Argument > Default
    server_url = os.getenv("SERVER_URL") or host or DEFAULT_URL
    server_port = os.getenv("SERVER_PORT") or port or DEFAULT_PORT
    if server_port == "auto":
        raise ValueError("Cannot use auto port on remote server")

    return ServerHandle(
        server_type=server_type_enum,
        deployment_type=deployment_type_enum,
        name=name,
        server_side_type=server_side_type,
        url=server_url,
        port=int(server_port),
    )


class Orchestra:
    @staticmethod
    def launch(
        # server information and deployment
        name: str | None = None,
        server_type: str | ServerType | None = None,
        deploy_to: str | None = None,
        server_side_type: str | None = None,
        # worker related inputs
        port: int | str | None = None,
        processes: int = 1,  # temporary work around for jax in subprocess
        dev_mode: bool = False,
        reset: bool = False,
        log_level: str | int | None = None,
        tail: bool = False,
        host: str | None = "0.0.0.0",  # nosec
        enable_warnings: bool = False,
        n_consumers: int = 0,
        thread_workers: bool = False,
        create_producer: bool = False,
        queue_port: int | None = None,
        association_request_auto_approval: bool = False,
        background_tasks: bool = False,
        debug: bool = False,
        migrate: bool = False,
        from_state_folder: str | Path | None = None,
        consumer_type: ConsumerType | None = None,
        db_url: str | None = None,
    ) -> ServerHandle:
        if from_state_folder is not None:
            with open(f"{from_state_folder}/config.json") as f:
                kwargs = json.load(f)
                server_handle = Orchestra.launch(**kwargs)
                client = server_handle.login(  # nosec
                    email="info@openmined.org", password="changethis"
                )
                client.load_migration_data(f"{from_state_folder}/migration.blob")
                return server_handle
        if dev_mode is True:
            thread_workers = True
        os.environ["DEV_MODE"] = str(dev_mode)

        server_type_enum: ServerType | None = get_server_type(server_type=server_type)
        server_side_type_enum = (
            ServerSideType.HIGH_SIDE
            if server_side_type is None
            else ServerSideType(server_side_type)
        )

        deployment_type_enum: DeploymentType | None = get_deployment_type(
            deployment_type=deploy_to
        )

        if deployment_type_enum == DeploymentType.PYTHON:
            server_handle = deploy_to_python(
                server_type_enum=server_type_enum,
                deployment_type_enum=deployment_type_enum,
                port=port,
                name=name,
                host=host,
                reset=reset,
                tail=tail,
                dev_mode=dev_mode,
                processes=processes,
                server_side_type=server_side_type_enum,
                enable_warnings=enable_warnings,
                log_level=log_level,
                n_consumers=n_consumers,
                thread_workers=thread_workers,
                create_producer=create_producer,
                queue_port=queue_port,
                association_request_auto_approval=association_request_auto_approval,
                background_tasks=background_tasks,
                debug=debug,
                migrate=migrate,
                consumer_type=consumer_type,
                db_url=db_url,
            )
            display(
                SyftInfo(
                    message=f"You have launched a development server at http://{host}:{server_handle.port}."
                    + " It is intended only for local use."
                )
            )
            return server_handle
        elif deployment_type_enum == DeploymentType.REMOTE:
            return deploy_to_remote(
                server_type_enum=server_type_enum,
                deployment_type_enum=deployment_type_enum,
                name=name,
                host=host,
                port=port,
                server_side_type=server_side_type_enum,
                migrate=migrate,
            )
        raise NotImplementedError(
            f"deployment_type: {deployment_type_enum} is not supported"
        )
