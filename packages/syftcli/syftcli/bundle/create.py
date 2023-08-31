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
from ..core.syft_version import SyftVersion

DEFAULT_OUTPUT_DIR = Path("~/.syft")


class ContainerEngineType(str, Enum):
    Docker = "docker"
    Podman = "podman"


def create(
    version: str = "latest",
    outdir: Annotated[
        Path, Option(dir_okay=True, file_okay=False, writable=True)
    ] = DEFAULT_OUTPUT_DIR,
    engine: ContainerEngineType = ContainerEngineType.Docker,
    dryrun: bool = False,
) -> None:
    """Create an offline container image bundle for a given Syft version."""

    # Validate Syft version. Fails if version is not valid.
    ver = validate_version(version)

    # Prepare output directory
    outdir = outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    tarpath = Path(outdir, f"syft-{ver.release_tag}-{engine}.tar")

    # Get container engine
    engine_sdk = get_container_engine(engine)

    # Begin bundling
    print(
        f"[bold green]Creating Syft {ver.release_tag} image bundle at '{tarpath}' using '{engine}'"
    )

    stream_output = {
        "cb_stdout": fn_print_std,
        "cb_stderr": fn_print_std,
    }

    images = get_syft_images(ver)

    print("\n[bold cyan]Pulling images...")
    engine_sdk.pull(images, stream_output=stream_output, dryrun=dryrun)

    print("\n[bold cyan]Creating tarball...")
    engine_sdk.save(images, tarfile=tarpath, dryrun=dryrun)

    print("\n[bold green]Done!")


def fn_print_std(line: str) -> None:
    print(f"[bright_black]{line}", end="", sep="")


def validate_version(version: str) -> SyftVersion:
    _ver: SyftVersion

    try:
        _ver = SyftVersion(version)
    except Exception:
        print(f"[bold red]'Error: {version}' is not a valid version")
        raise Exit(1)

    if not _ver.match(">0.8.1"):
        print("[bold red]Error: Minimum supported version is 0.8.2")
        raise Exit(1)

    if not _ver.valid_version():
        print(f"[bold red]Error: Version '{_ver}' is not a valid Syft release")
        raise Exit(1)

    return _ver


def get_container_engine(engine_name: ContainerEngineType) -> ContainerEngine:
    engine: ContainerEngine

    if engine_name == ContainerEngineType.Docker:
        engine = Docker()
    elif engine_name == ContainerEngineType.Podman:
        engine = Podman()

    if not engine.is_installed():
        print(f"[bold red]'{engine_name}' is not running or not installed")
        raise Exit(1)

    return engine


def get_syft_images(syft_ver: SyftVersion) -> List[str]:
    # TODO: manifest file in releases does not have these values yet!
    return [
        f"docker.io/openmined/grid-frontend:{syft_ver.docker_tag}",
        f"docker.io/openmined/grid-backend:{syft_ver.docker_tag}",
        # f"docker.io/openmined/grid-node-jupyter:{syft_ver.docker_tag}",
        "docker.io/library/mongo:7",
        "docker.io/traefik:v2.8.1",
    ]
