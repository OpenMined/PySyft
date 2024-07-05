# stdlib
from functools import cached_property
import os.path
from pathlib import Path
from typing import Any

# relative
from .builder_docker import DockerBuilder
from .builder_k8s import KubernetesBuilder
from .builder_types import BuilderBase
from .builder_types import ImageBuildResult
from .builder_types import ImagePushResult
from .config import CustomWorkerConfig
from .config import DockerWorkerConfig
from .config import WorkerConfig
from .k8s import IN_KUBERNETES

__all__ = ["CustomWorkerBuilder"]


class CustomWorkerBuilder:
    TYPE_CPU = "cpu"
    TYPE_GPU = "gpu"

    TEMPLATE_DIR_PROD = os.path.expandvars("$APPDIR/grid/images/")
    TEMPLATE_DIR_DEV = "../../../../../grid/backend/grid/images/"

    CUSTOM_IMAGE_PREFIX = "custom-worker"

    BUILD_MAX_WAIT = 30 * 60

    @cached_property
    def builder(self) -> BuilderBase:
        if IN_KUBERNETES:
            return KubernetesBuilder()
        else:
            return DockerBuilder()

    def build_image(
        self,
        config: WorkerConfig,
        tag: str | None = None,
        **kwargs: Any,
    ) -> ImageBuildResult:
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

    def push_image(
        self,
        tag: str,
        registry_url: str,
        username: str,
        password: str,
        **kwargs: Any,
    ) -> ImagePushResult:
        """
        Pushes a Docker image to the given repo.
        Args:
            repo (str): The repo to push the image to.
            tag (str): The tag to use for the image.
        """

        return self.builder.push_image(
            tag=tag,
            username=username,
            password=password,
            registry_url=registry_url,
        )

    def _build_dockerfile(
        self,
        config: DockerWorkerConfig,
        tag: str,
        **kwargs: Any,
    ) -> ImageBuildResult:
        return self.builder.build_image(
            dockerfile=config.dockerfile,
            tag=tag,
            **kwargs,
        )

    def _build_template(
        self,
        config: CustomWorkerConfig,
        **kwargs: Any,
    ) -> ImageBuildResult:
        # Builds a Docker pre-made CPU/GPU image template using a CustomWorkerConfig
        # remove once GPU is supported
        if config.build.gpu:
            raise Exception("GPU custom worker is not supported yet")

        type = self.TYPE_GPU if config.build.gpu else self.TYPE_CPU

        dockerfile_path = self.find_worker_image(type)

        imgtag = config.get_signature()[:8]

        build_args = {
            "PYTHON_VERSION": config.build.python_version,
            "SYSTEM_PACKAGES": config.build.merged_system_pkgs(),
            "PIP_PACKAGES": config.build.merged_python_pkgs(),
            "CUSTOM_CMD": config.build.merged_custom_cmds(),
        }

        return self.builder.build_image(
            tag=f"{self.CUSTOM_IMAGE_PREFIX}-{type}:{imgtag}",
            dockerfile_path=dockerfile_path,
            buildargs=build_args,
        )

    def find_worker_image(self, type: str) -> Path:
        """
        Find the Worker Dockerfile and it's context path
        - PROD will be in `$APPDIR/grid/`
        - DEV will be in `packages/grid/backend/grid/images`
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
                return path

        raise FileNotFoundError(f"Cannot find the {filename}")
