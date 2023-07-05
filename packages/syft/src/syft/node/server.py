# stdlib
import asyncio
from enum import Enum
import logging
import multiprocessing
import os
import platform
import signal
import subprocess  # nosec
import time
from typing import Callable
from typing import List
from typing import Tuple

# third party
from fastapi import APIRouter
from fastapi import FastAPI
import requests
from starlette.middleware.cors import CORSMiddleware
import uvicorn

# relative
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
    name: str, node_type: Enum, host: str, port: int, reset: bool, dev_mode: bool
):
    async def _run_uvicorn(
        name: str, node_type: Enum, host: str, port: int, reset: bool, dev_mode: bool
    ):
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
                processes=0,
                reset=reset,
                local_db=True,
                node_type=node_type,
            )
        else:
            worker = worker_class(
                name=name,
                processes=0,
                local_db=True,
                node_type=node_type,
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
    loop.run_until_complete(_run_uvicorn(name, node_type, host, port, reset, dev_mode))
    loop.close()


def serve_node(
    name: str,
    node_type: NodeType = NodeType.DOMAIN,
    host: str = "0.0.0.0",  # nosec
    port: int = 8080,
    reset: bool = False,
    dev_mode: bool = False,
    tail: bool = False,
) -> Tuple[Callable, Callable]:
    server_process = multiprocessing.Process(
        target=run_uvicorn, args=(name, node_type, host, port, reset, dev_mode)
    )

    def stop():
        print(f"Stopping {name}")
        server_process.terminate()
        server_process.join()

    def start():
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


def find_python_processes_on_port(port: int) -> List[int]:
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
