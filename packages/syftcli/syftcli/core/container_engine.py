# stdlib
from abc import ABC
from abc import abstractmethod

# third party
from rich.progress import track

# relative
from .proc import CalledProcessError
from .proc import CompletedProcess
from .proc import run_command


class ContainerEngineError(CalledProcessError):
    pass


class ContainerEngine(ABC):
    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def pull(
        self, images: list[str], dryrun: bool, stream_output: dict | None
    ) -> list[CompletedProcess]:
        raise NotImplementedError()

    @abstractmethod
    def save(
        self, images: list[str], archive_path: str, dryrun: bool
    ) -> CompletedProcess:
        raise NotImplementedError()

    def check_returncode(self, result: CompletedProcess) -> None:
        try:
            result.check_returncode()
        except CalledProcessError as e:
            raise ContainerEngineError(e.returncode, e.cmd) from e


class Podman(ContainerEngine):
    def is_available(self) -> bool:
        result = run_command("podman version")
        return result.returncode == 0

    def pull(
        self,
        images: list[str],
        dryrun: bool = False,
        stream_output: dict | None = None,
    ) -> list[CompletedProcess]:
        results = []

        for image in track(images, description=""):
            command = f"podman pull {image} --quiet"
            result = run_command(command, stream_output=stream_output, dryrun=dryrun)
            self.check_returncode(result)
            results.append(result)

        return results

    def save(
        self,
        images: list[str],
        archive_path: str,
        dryrun: bool = False,
    ) -> CompletedProcess:
        # -m works only with --format=docker-archive
        images_str = " ".join(images)
        command = f"podman save -m -o {archive_path} {images_str}"
        result = run_command(command, dryrun=dryrun)
        self.check_returncode(result)
        return result


class Docker(ContainerEngine):
    def is_available(self) -> bool:
        result = run_command("docker version")
        return result.returncode == 0

    def pull(
        self,
        images: list[str],
        dryrun: bool = False,
        stream_output: dict | None = None,
    ) -> list[CompletedProcess]:
        results = []

        for image in track(images, description=""):
            command = f"docker pull {image} --quiet"
            result = run_command(command, stream_output=stream_output, dryrun=dryrun)
            self.check_returncode(result)
            results.append(result)

        return results

    def save(
        self,
        images: list[str],
        archive_path: str,
        dryrun: bool = False,
    ) -> CompletedProcess:
        images_str = " ".join(images)
        command = f"docker save -o {archive_path} {images_str}"
        result = run_command(command, dryrun=dryrun)
        self.check_returncode(result)
        return result
