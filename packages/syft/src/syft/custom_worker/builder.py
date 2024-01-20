# stdlib
import contextlib
import io
import os.path
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Optional
from typing import Tuple

# third party
import docker
from docker.models.images import Image

# relative
from .config import CustomWorkerConfig
from .config import DockerWorkerConfig
from .config import WorkerConfig


class CustomWorkerBuilder:
    TYPE_CPU = "cpu"
    TYPE_GPU = "gpu"

    TEMPLATE_DIR_PROD = os.path.expandvars("$APPDIR/grid/")
    TEMPLATE_DIR_DEV = "../../../../../grid/backend/"

    CUSTOM_IMAGE_PREFIX = "custom-worker"

    BUILD_MAX_WAIT = 30 * 60

    def build_image(
        self,
        config: WorkerConfig,
        tag: str = None,
        **kwargs: Any,
    ) -> Tuple[Image, Iterable[str]]:
        """
        Builds a Docker image from the given configuration.
        Args:
            config (WorkerConfig): The configuration for building the Docker image.
            tag (str): The tag to use for the image.
        """

        if isinstance(config, DockerWorkerConfig):
            return self._build_dockerfile(config, tag, **kwargs)
        elif isinstance(config, CustomWorkerConfig):
            return self._build_template(config, **kwargs)
        else:
            raise TypeError("Unknown worker config type")

    def push_image(self, tag: str, **kwargs: Any) -> str:
        """
        Pushes a Docker image to the given repo.
        Args:
            repo (str): The repo to push the image to.
            tag (str): The tag to use for the image.
        """

        return self._push_image(tag, **kwargs)

    def _build_dockerfile(self, config: DockerWorkerConfig, tag: str, **kwargs):
        # convert string to file-like object
        file_obj = io.BytesIO(config.dockerfile.encode("utf-8"))
        return self._build_image(fileobj=file_obj, tag=tag, **kwargs)

    def _build_template(self, config: CustomWorkerConfig, **kwargs: Any):
        # Builds a Docker pre-made CPU/GPU image template using a CustomWorkerConfig
        # remove once GPU is supported
        if config.build.gpu:
            raise Exception("GPU custom worker is not supported yet")

        type = self.TYPE_GPU if config.build.gpu else self.TYPE_CPU

        contextdir, dockerfile = self._find_template_dir(type)

        imgtag = config.get_signature()[:8]

        build_args = {
            "PYTHON_VERSION": config.build.python_version,
            "SYSTEM_PACKAGES": config.build.merged_system_pkgs(),
            "PIP_PACKAGES": config.build.merged_python_pkgs(),
            "CUSTOM_CMD": config.build.merged_custom_cmds(),
        }

        return self._build_image(
            tag=f"{self.CUSTOM_IMAGE_PREFIX}-{type}:{imgtag}",
            path=str(contextdir),
            dockerfile=dockerfile,
            buildargs=build_args,
        )

    def _build_image(self, tag: str, **build_opts) -> Tuple[Image, Iterable]:
        # Core docker build call. Func signature should match with Docker SDK's BuildApiMixin
        with contextlib.closing(docker.from_env()) as client:
            image_result = client.images.build(
                tag=tag,
                timeout=self.BUILD_MAX_WAIT,
                **build_opts,
            )
            return image_result

    def _push_image(
        self,
        tag: str,
        registry_url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> str:
        with contextlib.closing(docker.from_env()) as client:
            if registry_url and username and password:
                client.login(
                    username=username, password=password, registry=registry_url
                )

            result = client.images.push(repository=tag)
            return result

    def _find_template_dir(self, type: str) -> Tuple[Path, str]:
        """
        Find the Worker Dockerfile and it's context path
        - PROD will be in `$APPDIR/grid/`
        - DEV will be in `packages/grid/backend`
        - In both the cases context dir does not matter (unless we're calling COPY)

        Args:
            type (str): The type of worker.
        Returns:
            Path: The path to the Dockerfile.
        """
        filename = f"worker_{type}.dockerfile"
        lookup_paths = [
            Path(self.TEMPLATE_DIR_PROD, filename).resolve(),
            Path(__file__, self.TEMPLATE_DIR_DEV, filename).resolve(),
        ]
        for path in lookup_paths:
            if path.exists():
                return path.parent, filename

        raise FileNotFoundError(f"Cannot find the {filename}")
