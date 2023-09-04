# stdlib
from enum import Enum
from pathlib import Path
from shutil import rmtree
import tarfile
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
from ..core.syft_repo import SyftRepo
from ..core.syft_version import InvalidVersion
from ..core.syft_version import SyftVersion

__all__ = "create"

DEFAULT_OUTPUT_DIR = Path("~/.syft")


class ContainerEngineType(str, Enum):
    Docker = "docker"
    Podman = "podman"


def create(
    version: Annotated[str, Option("--version", "-v")] = "latest",
    outdir: Annotated[
        Path, Option("--outdir", "-d", dir_okay=True, file_okay=False, writable=True)
    ] = DEFAULT_OUTPUT_DIR,
    engine: Annotated[
        ContainerEngineType, Option("--engine", "-e")
    ] = ContainerEngineType.Docker,
    dryrun: bool = False,
) -> None:
    """Create an offline deployment bundle for Syft."""

    # Validate Syft version. Fails if version is not valid.
    ver = validate_version(version)

    # Prepare temp paths
    out_path = prepare_output_dir(outdir)
    temp_path = prepare_tmp_dir(out_path)
    img_path = Path(temp_path, "images.tar")

    # prepare output paths
    bundle_path = Path(out_path, f"syft-{ver.release_tag}-{engine.value}.tar")

    # Prepare container engine & images
    engine_sdk = get_container_engine(engine)
    image_tags = get_syft_images(ver)

    # Begin bundling
    print(
        f"[bold green]"
        f"Creating Syft {ver.release_tag} {engine} bundle at '{bundle_path}'"
    )

    print("\n[bold cyan]Pulling images...")
    engine_sdk.pull(
        image_tags,
        stream_output={"cb_stdout": fn_print_std, "cb_stderr": fn_print_std},
        dryrun=dryrun,
    )

    print("\n[bold cyan]Creating image archive...")
    engine_sdk.save(image_tags, tarfile=img_path, dryrun=dryrun)

    print(f"\n[bold cyan]Downloading {engine} config...")
    asset_path = get_engine_config(engine, ver, temp_path, dryrun=dryrun)

    print("\n[bold cyan]Creating final bundle...")
    create_syft_bundle(bundle_path, images=img_path, assets=asset_path, dryrun=dryrun)

    print("\n[bold cyan]Cleaning up...")
    cleanup_dir(temp_path)

    print("\n[bold green]Done!")


def fn_print_std(line: str) -> None:
    print(f"[bright_black]{line}", end="", sep="")


def validate_version(version: str) -> SyftVersion:
    _ver: SyftVersion

    try:
        _ver = SyftVersion(version)
    except InvalidVersion:
        print(f"[bold red]Error: '{version}' is not a valid version")
        raise Exit(1)

    if _ver.match("<0.8.2b27"):
        print(f"[bold red]Error: Minimum supported version is 0.8.2. Got: {_ver}")
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
    manifest = SyftRepo.get_manifest(syft_ver.release_tag)
    return manifest["images"]


def prepare_output_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_tmp_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    path = path / "temp"

    if path.exists():
        rmtree(path)

    path.mkdir(parents=True, exist_ok=True)

    return path


def cleanup_dir(path: Path) -> None:
    if path.exists():
        rmtree(path)


def get_engine_config(
    engine: ContainerEngineType,
    ver: SyftVersion,
    dl_dir: Path,
    dryrun: bool = False,
) -> Path:
    asset_name = (
        SyftRepo.Assets.PODMAN_CONFIG
        if engine == ContainerEngineType.Podman
        else SyftRepo.Assets.DOCKER_CONFIG
    )

    if dryrun:
        return Path(dl_dir, asset_name)

    return SyftRepo.download_asset(asset_name, ver.release_tag, dl_dir)


def create_syft_bundle(
    path: Path,
    images: Path,
    assets: Path,
    dryrun: bool = False,
) -> None:
    if dryrun:
        return

    if path.exists():
        path.unlink()

    with tarfile.open(str(path), "w") as bundle:
        # extract assets as-is into bundle root
        with tarfile.open(str(assets), "r:gz") as asset:
            for member in asset.getmembers():
                bundle.addfile(member, asset.extractfile(member))

        # add images archive into the bundle
        bundle.add(images, arcname=images.name)
