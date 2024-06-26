# stdlib
import os
import subprocess

# third party
import uvicorn

# relative
from ..abstract_node import NodeSideType

# from .server import app_factory
from ..orchestra import DeploymentType
from ..orchestra import NodeHandle
from .node import NodeType

# List storing all reloadable servers launched
process_list = []

def start_reloadable_server(
    name: str = "testing-node",
    node_type: str = "domain",
    node_side_type: str = "high",
    port: int = 9081,
    processes: int = 1,
    local_db: bool = True,
    reset: bool = False,
) -> NodeHandle:
    os.environ["NODE_NAME"] = name
    os.environ["NODE_TYPE"] = node_type
    os.environ["NODE_SIDE_TYPE"] = node_side_type
    os.environ["PORT"] = str(port)
    os.environ["PROCESSES"] = str(processes)
    os.environ["LOCAL_DB"] = str(local_db)
    os.environ["RESET"] = str(reset)

    command = ["python", "-m", "syft.node.server_management"]
    process = subprocess.Popen(command)
    process_list.append(process)
    print("*" * 50, flush=True)
    print(f"Uvicorn server running on port {port} with PID: {process.pid}", flush=True)
    print("*" * 50, flush=True)

    
    # Since the servers take a second to run, adding this wait so
    # that notebook commands can run one after the other.
    # stdlib
    from time import sleep 
    sleep(6)
    
    def stop() -> None:
        process.terminate()
        process.wait()
        if process in process_list:
            process_list.remove(process)
        print("*" * 50, flush=True)
        print(f"Uvicorn server with PID: {process.pid} stopped.", flush=True)
        print("*" * 50, flush=True)

        
    # Return this object:
    return NodeHandle(
        node_type=NodeType(node_type),
        deployment_type=DeploymentType.PYTHON,
        name=name,
        port=port,
        url="http://localhost",
        node_side_type=NodeSideType(node_side_type),
        shutdown=stop,
    )

def stop_all_reloadable_servers() -> None:
    for process in process_list:
        process.terminate()
        process.wait()
    process_list.clear()
    print("All Uvicorn servers stopped.")


if __name__ == "__main__":
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    reload_dirs = os.path.abspath(os.path.join(current_file_path, "../../"))

    uvicorn.run(
        "syft.node.server:app_factory",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 9081)),
        reload=True,
        factory=True,
        reload_dirs=[reload_dirs],
    )