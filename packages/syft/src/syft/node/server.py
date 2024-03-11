# stdlib
import asyncio
from collections.abc import Callable
from enum import Enum
import logging
import multiprocessing
import os
import platform
import signal
import subprocess  # nosec
import time

# third party
from fastapi import APIRouter
from fastapi import FastAPI
import requests
from starlette.middleware.cors import CORSMiddleware
import uvicorn

# relative
from ..abstract_node import NodeSideType
from ..client.client import API_PATH
from ..util.constants import DEFAULT_TIMEOUT
from ..util.util import os_name
from .domain import Domain
from .enclave import Enclave
from .gateway import Gateway
from .node import NodeType
from .routes import make_routes

if os_name() == "macOS":
    # needed on MacOS to prevent [__NSCFConstantString initialize] may have been in
    # progress in another thread when fork() was called.
    multiprocessing.set_start_method("spawn", True)

WAIT_TIME_SECONDS = 20


def make_app(name: str, router: APIRouter) -> FastAPI:
    app = FastAPI(
        title=name,
    )

    api_router = APIRouter()

    api_router.include_router(router)
    app.include_router(api_router, prefix="/api/v2")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


worker_classes = {
    NodeType.DOMAIN: Domain,
    NodeType.GATEWAY: Gateway,
    NodeType.ENCLAVE: Enclave,
}


def run_uvicorn(
    name: str,
    node_type: Enum,
    host: str,
    port: int,
    processes: int,
    reset: bool,
    dev_mode: bool,
    node_side_type: str,
    enable_warnings: bool,
    in_memory_workers: bool,
    queue_port: int | None,
    create_producer: bool,
    n_consumers: int,
) -> None:
    async def _run_uvicorn(
        name: str,
        node_type: NodeType,
        host: str,
        port: int,
        reset: bool,
        dev_mode: bool,
        node_side_type: Enum,
    ) -> None:
        if node_type not in worker_classes:
            raise NotImplementedError(f"node_type: {node_type} is not supported")
        worker_class = worker_classes[node_type]
        if dev_mode:
            print(
                f"\nWARNING: private key is based on node name: {name} in dev_mode. "
                "Don't run this in production."
            )

            worker = worker_class.named(
                name=name,
                processes=processes,
                reset=reset,
                local_db=True,
                node_type=node_type,
                node_side_type=node_side_type,
                enable_warnings=enable_warnings,
                migrate=True,
                in_memory_workers=in_memory_workers,
                queue_port=queue_port,
                create_producer=create_producer,
                n_consumers=n_consumers,
            )
        else:
            worker = worker_class(
                name=name,
                processes=processes,
                local_db=True,
                node_type=node_type,
                node_side_type=node_side_type,
                enable_warnings=enable_warnings,
                migrate=True,
                in_memory_workers=in_memory_workers,
                queue_port=queue_port,
                create_producer=create_producer,
                n_consumers=n_consumers,
            )
        router = make_routes(worker=worker)
        app = make_app(worker.name, router=router)

        if reset:
            try:
                python_pids = find_python_processes_on_port(port)
                for pid in python_pids:
                    print(f"Stopping process on port: {port}")
                    kill_process(pid)
                    time.sleep(1)
            except Exception:  # nosec
                print(f"Failed to kill python process on port: {port}")

        log_level = "critical"
        if dev_mode:
            log_level = "info"
            logging.getLogger("uvicorn").setLevel(logging.CRITICAL)
            logging.getLogger("uvicorn.access").setLevel(logging.CRITICAL)
        config = uvicorn.Config(
            app, host=host, port=port, log_level=log_level, reload=dev_mode
        )
        server = uvicorn.Server(config)

        await server.serve()
        asyncio.get_running_loop().stop()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(
        _run_uvicorn(
            name,
            node_type,
            host,
            port,
            reset,
            dev_mode,
            node_side_type,
        )
    )
    loop.close()


def serve_node(
    name: str,
    node_type: NodeType = NodeType.DOMAIN,
    node_side_type: NodeSideType = NodeSideType.HIGH_SIDE,
    host: str = "0.0.0.0",  # nosec
    port: int = 8080,
    processes: int = 1,
    reset: bool = False,
    dev_mode: bool = False,
    tail: bool = False,
    enable_warnings: bool = False,
    in_memory_workers: bool = True,
    queue_port: int | None = None,
    create_producer: bool = False,
    n_consumers: int = 0,
) -> tuple[Callable, Callable]:
    server_process = multiprocessing.Process(
        target=run_uvicorn,
        args=(
            name,
            node_type,
            host,
            port,
            processes,
            reset,
            dev_mode,
            node_side_type,
            enable_warnings,
            in_memory_workers,
            queue_port,
            create_producer,
            n_consumers,
        ),
    )

    def stop() -> None:
        print(f"Stopping {name}")
        server_process.terminate()
        server_process.join(3)
        if server_process.is_alive():
            # this is needed because often the process is still alive
            server_process.kill()
            print("killed")

    def start() -> None:
        print(f"Starting {name} server on {host}:{port}")
        server_process.start()

        if tail:
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                try:
                    stop()
                except SystemExit:
                    os._exit(130)
        else:
            for i in range(WAIT_TIME_SECONDS):
                try:
                    req = requests.get(
                        f"http://{host}:{port}{API_PATH}/metadata",
                        timeout=DEFAULT_TIMEOUT,
                    )
                    if req.status_code == 200:
                        print(" Done.")
                        break
                except Exception:
                    time.sleep(1)
                    if i == 0:
                        print("Waiting for server to start", end="")
                    else:
                        print(".", end="")

    return start, stop


def find_python_processes_on_port(port: int) -> list[int]:
    system = platform.system()

    if system == "Windows":
        command = f"netstat -ano | findstr :{port}"
        process = subprocess.Popen(  # nosec
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        output, _ = process.communicate()
        pids = [
            int(line.strip().split()[-1]) for line in output.split("\n") if line.strip()
        ]

    else:  # Linux and MacOS
        command = f"lsof -i :{port} -sTCP:LISTEN -t"
        process = subprocess.Popen(  # nosec
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        output, _ = process.communicate()
        pids = [int(pid.strip()) for pid in output.split("\n") if pid.strip()]

    python_pids = []
    for pid in pids:
        try:
            if system == "Windows":
                command = (
                    f"wmic process where (ProcessId='{pid}') get ProcessId,CommandLine"
                )
            else:
                command = f"ps -p {pid} -o pid,command"

            process = subprocess.Popen(  # nosec
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            output, _ = process.communicate()
            lines = output.strip().split("\n")

            if len(lines) > 1 and "python" in lines[1].lower():
                python_pids.append(pid)

        except Exception as e:
            print(f"Error checking process {pid}: {e}")

    return python_pids


def kill_process(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Process {pid} terminated.")
    except Exception as e:
        print(f"Error killing process {pid}: {e}")
