# future
from __future__ import annotations

# stdlib
from enum import Enum
from pathlib import Path
from typing import List

# third party
from rich import print
from typer import Exit
from typer import Option
from typing_extensions import Annotated

# relative
from ..core.container_engine import ContainerEngine
from ..core.container_engine import Docker
from ..core.container_engine import Podman

LATEST_VERSION = "0.8.2-beta.6"
DEFAULT_TAR_FILE = Path("./syft_images/package.tar")


class ContainerEngineType(str, Enum):
    Docker = "docker"
    Podman = "podman"


def create(
    version: str = "latest",
    tarfile: Annotated[Path, Option(dir_okay=True, writable=True)] = DEFAULT_TAR_FILE,
    engine: ContainerEngineType = ContainerEngineType.Docker,
    dryrun: bool = False,
) -> None:
    if not verify_syft_version(version):
        raise Exit(1)

    engine_sdk = get_container_engine(engine)
    if not engine_sdk.is_installed():
        print(f"[bold red]{engine} is not installed")
        raise Exit(1)

    print(
        f"[bold green]Creating Syft {version} image bundle at '{tarfile.resolve()}' using {engine}"
    )

    tarfile.parent.mkdir(parents=True, exist_ok=True)
    images = images_for_syft_version(version)
    stream_output = {
        "cb_stdout": print_std,
        "cb_stderr": print_std,
    }

    print("\n[bold cyan]Pulling images...")
    engine_sdk.pull(images, stream_output=stream_output, dryrun=dryrun)

    print("\n[bold cyan]Creating tarball...")
    engine_sdk.save(images, tarfile=tarfile, dryrun=dryrun)

    print("\n[bold green] Done!")


def print_std(line: str) -> None:
    print(f"[bright_black]{line}", end="", sep="")


def get_container_engine(engine_name: ContainerEngineType) -> ContainerEngine:
    if engine_name == ContainerEngineType.Docker:
        return Docker()
    elif engine_name == ContainerEngineType.Podman:
        return Podman()

    raise ValueError(f"{engine_name} is not a valid container engine")


def verify_syft_version(version: str) -> bool:
    # TODO: implement this
    return True


def images_for_syft_version(version: str = "latest") -> List[str]:
    # TODO: Get this dynamically by version
    if version == "latest":
        version = LATEST_VERSION

    return [
        f"docker.io/openmined/grid-frontend:{version}",
        f"docker.io/openmined/grid-backend:{version}",
        f"docker.io/openmined/grid-node-jupyter:{version}",
        "docker.io/library/mongo:7",
        "docker.io/traefik:v2.8.1",
    ]
