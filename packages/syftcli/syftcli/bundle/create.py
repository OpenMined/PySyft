# stdlib
from enum import Enum
from pathlib import Path
from shutil import rmtree
import tarfile
from typing import Annotated

# third party
from typer import Exit
from typer import Option

# relative
from ..core.console import debug
from ..core.console import error
from ..core.console import info
from ..core.console import success
from ..core.container_engine import ContainerEngine
from ..core.container_engine import ContainerEngineError
from ..core.container_engine import Docker
from ..core.container_engine import Podman
from ..core.syft_repo import SyftRepo
from ..core.syft_version import InvalidVersion
from ..core.syft_version import SyftVersion

__all__ = "create"

DEFAULT_OUTPUT_DIR = Path("~/.syft")


class Engine(str, Enum):
    Docker = "docker"
    Podman = "podman"


VersionOpts = Annotated[str, Option("--version", "-v")]
EngineOpts = Annotated[Engine, Option("--engine", "-e")]
DryrunOpts = Annotated[bool, Option("--dryrun")]
OutdirOpts = Annotated[
    Path, Option("--outdir", "-d", dir_okay=True, file_okay=False, writable=True)
]


def create(
    version: VersionOpts = "latest",
    outdir: OutdirOpts = DEFAULT_OUTPUT_DIR,
    engine: EngineOpts = Engine.Docker,
    dryrun: DryrunOpts = False,
) -> None:
    """Create an offline deployment bundle for Syft."""

    # Validate Syft version. Fails if version is not valid.
    ver = validate_version(version)

    # Prepare temp paths
    out_path = prepare_output_dir(outdir)
    temp_path = prepare_tmp_dir(out_path)
    archive_path = Path(temp_path, "images.tar")

    # prepare output paths
    bundle_path = Path(out_path, f"syft-{ver.release_tag}-{engine.value}.tar")

    # Prepare container engine & images
    engine_sdk = get_container_engine(engine, dryrun=dryrun)
    image_tags = get_syft_images(ver)

    # Begin bundling
    info(f"Creating Syft {ver.release_tag} {engine} bundle at '{bundle_path}'")

    info("\nPulling images...")
    pull_images(engine_sdk, image_tags, dryrun=dryrun)

    info("\nCreating image archive...")
    archive_images(engine_sdk, image_tags, archive_path, dryrun=dryrun)

    info(f"\nDownloading {engine.value} config...")
    config_path = get_engine_config(engine, ver, temp_path, dryrun=dryrun)

    info("\nCreating final bundle...")
    create_syft_bundle(bundle_path, archive_path, config_path, dryrun=dryrun)

    info("\nCleaning up...")
    cleanup_dir(temp_path)

    success("\nDone!")


def validate_version(version: str) -> SyftVersion:
    _ver: SyftVersion

    try:
        _ver = SyftVersion(version)
    except InvalidVersion:
        error(f"Error: '{version}' is not a valid version")
        raise Exit(1)

    if _ver.match("<0.8.2b27"):
        error(f"Error: Minimum supported version is '0.8.2' Got: '{_ver}'")
        raise Exit(1)

    if not _ver.valid_version():
        error(f"Error: Version '{_ver}' is not a valid Syft release")
        raise Exit(1)

    return _ver


def get_container_engine(engine_name: Engine, dryrun: bool = False) -> ContainerEngine:
    engine: ContainerEngine

    if engine_name == Engine.Docker:
        engine = Docker()
    elif engine_name == Engine.Podman:
        engine = Podman()

    if not dryrun and not engine.is_available():
        error(
            f"Error: '{engine_name}' is unavailable. "
            "Make sure it is installed and running."
        )
        raise Exit(1)

    return engine


def pull_images(
    engine_sdk: ContainerEngine,
    image_tags: list[str],
    dryrun: bool = False,
) -> None:
    def fn_print_std(line: str) -> None:
        debug(line, end="", sep="")

    try:
        results = engine_sdk.pull(
            image_tags,
            stream_output={"cb_stdout": fn_print_std, "cb_stderr": fn_print_std},
            dryrun=dryrun,
        )
        dryrun and [debug(result.args) for result in results]  # type: ignore[func-returns-value]
    except ContainerEngineError as e:
        error("Error:", e)
        raise Exit(e.returncode)


def archive_images(
    engine_sdk: ContainerEngine,
    image_tags: list[str],
    archive_path: Path,
    dryrun: bool = False,
) -> None:
    try:
        result = engine_sdk.save(image_tags, archive_path, dryrun=dryrun)
        dryrun and debug(result.args)  # type: ignore[func-returns-value]
    except ContainerEngineError as e:
        error("Error:", e)
        raise Exit(e.returncode)


def get_syft_images(syft_ver: SyftVersion) -> list[str]:
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
    engine: Engine,
    ver: SyftVersion,
    dl_dir: Path,
    dryrun: bool = False,
) -> Path:
    asset_name = (
        SyftRepo.Assets.PODMAN_CONFIG
        if engine == Engine.Podman
        else SyftRepo.Assets.DOCKER_CONFIG
    )

    if dryrun:
        debug(f"Download: '{ver.release_tag}/{asset_name}' to '{dl_dir}'")
        return Path(dl_dir, asset_name)

    return SyftRepo.download_asset(asset_name, ver.release_tag, dl_dir)


def create_syft_bundle(
    bundle_path: Path,
    archive_path: Path,
    config_path: Path,
    dryrun: bool = False,
) -> None:
    if dryrun:
        debug(
            f"Bundle: {bundle_path}\n"
            f"+ Image: {archive_path}\n"
            f"+ Deployment Config: {config_path}\n"
        )
        return

    if bundle_path.exists():
        bundle_path.unlink()

    with tarfile.open(str(bundle_path), "w") as bundle:
        # extract assets config as-is into bundle root
        with tarfile.open(str(config_path), "r:gz") as asset:
            for member in asset.getmembers():
                bundle.addfile(member, asset.extractfile(member))

        # add image archive into the bundle
        bundle.add(archive_path, arcname=archive_path.name)
