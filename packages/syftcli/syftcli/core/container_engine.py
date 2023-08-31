# stdlib
from typing import List
from typing import Optional

# third party
from rich.progress import track

# relative
from .proc import handle_error
from .proc import run_command


class ContainerEngine:
    pass


class Podman(ContainerEngine):
    def is_installed(self) -> bool:
        result = run_command("podman version")
        return result[-1] == 0

    def pull(
        self,
        images: List[str],
        dryrun: bool = False,
        stream_output: Optional[dict] = None,
    ) -> None:
        for image in track(images, description=""):
            command = f"podman pull {image} --quiet"

            if dryrun:
                print(command)
                continue

            result = run_command(command, stream_output=stream_output)
            handle_error(result)

    def save(self, images: List[str], tarfile: str, dryrun: bool = False) -> None:
        images_str = " ".join(images)
        command = f"podman save -o {tarfile} {images_str}"

        if dryrun:
            print(command)
            return

        result = run_command(command)
        handle_error(result)


class Docker(ContainerEngine):
    def is_installed(self) -> bool:
        result = run_command("docker version")
        return result[-1] == 0

    def pull(
        self,
        images: List[str],
        dryrun: bool = False,
        stream_output: Optional[dict] = None,
    ) -> None:
        for image in track(images, description=""):
            command = f"docker pull {image} --quiet"

            if dryrun:
                print(command)
                continue

            result = run_command(command, stream_output=stream_output)
            handle_error(result)

    def save(self, images: List[str], tarfile: str, dryrun: bool = False) -> None:
        images_str = " ".join(images)
        command = f"docker save -o {tarfile} {images_str}"

        if dryrun:
            print(command)
            return

        result = run_command(command)
        handle_error(result)
