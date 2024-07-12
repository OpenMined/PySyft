# stdlib
from collections.abc import Callable
import multiprocessing
import multiprocessing.synchronize
import os
from pathlib import Path
import platform
import signal
import subprocess  # nosec
import sys
import time
from typing import Any

# third party
from fastapi import APIRouter
from fastapi import FastAPI
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict
import requests
from starlette.middleware.cors import CORSMiddleware
import uvicorn

# relative
from ..abstract_node import NodeSideType
from ..client.client import API_PATH
from ..util.autoreload import enable_autoreload
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


class AppSettings(BaseSettings):
    name: str
    node_type: NodeType = NodeType.DOMAIN
    node_side_type: NodeSideType = NodeSideType.HIGH_SIDE
    processes: int = 1
    reset: bool = False
    dev_mode: bool = False
    enable_warnings: bool = False
    in_memory_workers: bool = True
    queue_port: int | None = None
    create_producer: bool = False
    n_consumers: int = 0
    association_request_auto_approval: bool = False
    background_tasks: bool = False
    payment_required: bool = False
    node_payment_handle: str | None = None
    payment_api: str | None = None

    model_config = SettingsConfigDict(env_prefix="SYFT_", env_parse_none_str="None")


def app_factory() -> FastAPI:
    settings = AppSettings()

    worker_classes = {
        NodeType.DOMAIN: Domain,
        NodeType.GATEWAY: Gateway,
        NodeType.ENCLAVE: Enclave,
    }
    if settings.node_type not in worker_classes:
        raise NotImplementedError(f"node_type: {settings.node_type} is not supported")
    worker_class = worker_classes[settings.node_type]

    kwargs = settings.model_dump()
    if settings.dev_mode:
        print(
            f"\nWARNING: private key is based on node name: {settings.name} in dev_mode. "
            "Don't run this in production."
        )
        worker = worker_class.named(**kwargs)
    else:
        del kwargs["reset"]  # Explicitly remove reset from kwargs for non-dev mode
        worker = worker_class(**kwargs)

    app = FastAPI(title=settings.name)
    router = make_routes(worker=worker)
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


def attach_debugger() -> None:
    # third party
    import debugpy

    os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
    _, debug_port = debugpy.listen(0)
    print(
        "\nStarting the server with the Python Debugger enabled (`debug=True`).\n"
        'To attach the debugger, open the command palette in VSCode and select "Debug: Start Debugging (F5)".\n'
        f"Then, enter `{debug_port}` in the port field and press Enter.\n",
        flush=True,
    )
    print(f"Waiting for debugger to attach on port `{debug_port}`...", flush=True)
    debugpy.wait_for_client()  # blocks execution until a remote debugger is attached
    print("Debugger attached", flush=True)


def run_uvicorn(
    host: str,
    port: int,
    starting_uvicorn_event: multiprocessing.synchronize.Event,
    **kwargs: Any,
) -> None:
    if kwargs.get("reset"):
        try:
            python_pids = find_python_processes_on_port(port)
            for pid in python_pids:
                print(f"Stopping process on port: {port}")
                kill_process(pid)
                time.sleep(1)
        except Exception:  # nosec
            print(f"Failed to kill python process on port: {port}")

    if kwargs.get("debug"):
        attach_debugger()

    # Set up all kwargs as environment variables so that they can be accessed in the app_factory function.
    env_prefix = AppSettings.model_config.get("env_prefix", "")
    for key, value in kwargs.items():
        key_with_prefix = f"{env_prefix}{key.upper()}"
        os.environ[key_with_prefix] = str(value)

    # The `serve_node` function calls `run_uvicorn` in a separate process using `multiprocessing.Process`.
    # When the child process is created, it inherits the file descriptors from the parent process.
    # If the parent process has a file descriptor open for sys.stdin, the child process will also have a file descriptor
    # open for sys.stdin. This can cause an OSError in uvicorn when it tries to access sys.stdin in the child process.
    # To prevent this, we set sys.stdin to None in the child process. This is safe because we don't actually need
    # sys.stdin while running uvicorn programmatically.
    sys.stdin = None  # type: ignore

    # Signal the parent process that we are starting the uvicorn server.
    starting_uvicorn_event.set()

    # Finally, run the uvicorn server.
    uvicorn.run(
        "syft.node.server:app_factory",
        host=host,
        port=port,
        factory=True,
        reload=kwargs.get("dev_mode"),
        reload_dirs=[Path(__file__).parent.parent] if kwargs.get("dev_mode") else None,
    )


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
    association_request_auto_approval: bool = False,
    background_tasks: bool = False,
    debug: bool = False,
    payment_required: bool = False,
    node_payment_handle: str | None = None,
    payment_api: str | None = None
) -> tuple[Callable, Callable]:
    starting_uvicorn_event = multiprocessing.Event()

    # Enable IPython autoreload if dev_mode is enabled.
    if dev_mode:
        enable_autoreload()

    server_process = multiprocessing.Process(
        target=run_uvicorn,
        kwargs={
            "name": name,
            "node_type": node_type,
            "host": host,
            "port": port,
            "processes": processes,
            "reset": reset,
            "dev_mode": dev_mode,
            "node_side_type": node_side_type,
            "enable_warnings": enable_warnings,
            "in_memory_workers": in_memory_workers,
            "queue_port": queue_port,
            "create_producer": create_producer,
            "n_consumers": n_consumers,
            "association_request_auto_approval": association_request_auto_approval,
            "background_tasks": background_tasks,
            "debug": debug,
            "payment_required": payment_required,
            "node_payment_handle": node_payment_handle,
            "payment_api": payment_api,
            "starting_uvicorn_event": starting_uvicorn_event,
        },
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

        # Wait for the child process to start uvicorn server before starting the readiness checks.
        starting_uvicorn_event.wait()

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
        if system == "Windows":
            command = (
                f"wmic process where (ProcessId='{pid}') get ProcessId,CommandLine"
            )
        else:
            command = f"ps -p {pid} -o pid,command"

        try:
            process = subprocess.Popen(  # nosec
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            output, _ = process.communicate()
        except Exception as e:
            print(f"Error checking process {pid}: {e}")
            continue

        lines = output.strip().split("\n")
        if len(lines) > 1 and "python" in lines[1].lower():
            python_pids.append(pid)

    return python_pids


def kill_process(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Process {pid} terminated.")
    except Exception as e:
        print(f"Error killing process {pid}: {e}")
