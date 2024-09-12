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
        """Returns the appropriate builder instance based on the environment.

        Returns:
            BuilderBase: An instance of either KubernetesBuilder or DockerBuilder.
        """
        if IN_KUBERNETES:
            return KubernetesBuilder()
        else:
            return DockerBuilder()

    def build_image(
        self,
        config: WorkerConfig,
        tag: str | None,
        **kwargs: Any,
    ) -> ImageBuildResult:
        """Builds a Docker image from the given configuration.

        Args:
            config (WorkerConfig): The configuration for building the Docker image.
            tag (str | None): The tag to use for the image. Defaults to None.
            **kwargs (Any): Additional keyword arguments for the build process.

        Returns:
            ImageBuildResult: The result of the image build process.

        Raises:
            TypeError: If the config type is not recognized.
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
        """Pushes a Docker image to the given registry.

        Args:
            tag (str): The tag of the image to push.
            registry_url (str): The URL of the registry.
            username (str): The username for registry authentication.
            password (str): The password for registry authentication.
            **kwargs (Any): Additional keyword arguments for the push process.

        Returns:
            ImagePushResult: The result of the image push process.
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
        """Builds a Docker image using a Dockerfile.

        Args:
            config (DockerWorkerConfig): The configuration containing the Dockerfile.
            tag (str): The tag to use for the image.
            **kwargs (Any): Additional keyword arguments for the build process.

        Returns:
            ImageBuildResult: The result of the image build process.
        """
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
        """Builds a Docker image using a pre-made template.

        Args:
            config (CustomWorkerConfig): The configuration containing template settings.
            **kwargs (Any): Additional keyword arguments for the build process.

        Returns:
            ImageBuildResult: The result of the image build process.

        Raises:
            Exception: If GPU support is requested but not supported.
        """
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
        """Finds the Worker Dockerfile and its context path.

        The production Dockerfile will be located at `$APPDIR/grid/`.
        The development Dockerfile will be located in `packages/grid/backend/grid/images`.

        Args:
            type (str): The type of worker (e.g., 'cpu' or 'gpu').

        Returns:
            Path: The path to the Dockerfile.

        Raises:
            FileNotFoundError: If the Dockerfile is not found in any of the expected locations.
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
