"""Python Level API to launch Docker Containers using Hagrid"""
# stdlib
import re
import subprocess  # nosec
from typing import Optional

# third party
import gevent

# relative
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


class Orchestra:
    def __init__(self, port: int, name: str):
        self.port = int(port)
        self.name = name

    @staticmethod
    def launch_worker(
        name: Optional[str] = None, dev_mode: bool = True, reset: bool = False
    ) -> "Orchestra":
        # Currently by default we launch in dev mode

        if reset:
            Orchestra.reset(name)

        # Start a subprocess and capture its output
        commands = ["hagrid", "launch"]

        name = random_name() if not name else name
        commands.extend([name, "enclave"])
        if dev_mode:
            commands.append("--dev")

        process = subprocess.Popen(
            commands,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Start gevent threads to read and print the output and error streams
        stdout_thread = gevent.spawn(read_stream, process.stdout)
        stderr_thread = gevent.spawn(read_stream, process.stderr)

        # Wait for the threads to finish
        gevent.joinall([stdout_thread, stderr_thread])

        return Orchestra.from_name(name=name)

    @staticmethod
    def from_name(name: str) -> "Orchestra":
        snake_name = to_snake_case(name)
        worker_containers = shell(
            "docker ps --format '{{.Names}} {{.Ports}}' | grep 'worker'"
        ).strip()

        if not worker_containers:
            raise Exception(
                f"Worker Container:{snake_name} not found"
                + "Kindly ensure the docker container is running"
            )

        _exists = False

        for worker_container in worker_containers.split("\n"):
            if snake_name in worker_container:
                _exists = True
                match = re.search(r"\d+(?=\->)", worker_container)
                if match:
                    port = match.group()
                else:
                    raise Exception(
                        f"Could not find port for Container: {worker_container}"
                    )
                break

        if not _exists:
            raise Exception(
                f"Worker Container:{snake_name} not found"
                + "Kindly ensure the docker container is running"
            )
        return Orchestra(port=port, name=name)

    def land(self) -> None:
        Orchestra.reset(self.name)

    @staticmethod
    def reset(name: str) -> None:
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
            print(f"❌ Unable to remove container volume: {snake_name} :{volume_output}")

    def __repr__(self) -> str:
        res = f"name: {self.name}"
        res += f"\nport: {self.port}"
        res += "\n\nKindly login using:"
        res += "\n\nimport syft as sy"
        res += "\nsy.login(port=..., email=..., password=...)"
        return res
