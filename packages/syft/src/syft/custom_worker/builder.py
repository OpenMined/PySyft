# stdlib
import os.path
from pathlib import Path

# third party
import docker

# relative
from .config import CustomWorkerConfig


class CustomWorkerBuilder:
    TYPE_CPU = "cpu"
    TYPE_GPU = "gpu"

    DOCKERFILE_PROD_PATH = os.path.expandvars("$APPDIR/grid/")
    DOCKERFILE_DEV_PATH = "../../../../../grid/backend/"

    CUSTOM_IMAGE_PREFIX = "custom-worker"

    BUILD_MAX_WAIT = 30 * 60

    def build_image(self, config: CustomWorkerConfig) -> bool:
        """
        Builds a Docker image for the custom worker based on the provided configuration.
        Args:
            config (CustomImageConfig): The configuration for building the Docker image.
        Returns:
            bool: True if the image was built successfully, raises Exception otherwise.
        """

        # remove once GPU is supported
        if config.build.gpu:
            raise Exception("GPU custom worker is not supported yet")

        type = self.TYPE_GPU if config.build.gpu else self.TYPE_CPU

        contextdir, dockerfile = self.find_worker_ctx(type)

        imgtag = config.get_signature()[:8]

        build_args = {
            "PYTHON_VERSION": config.build.python_version,
            "SYSTEM_PACKAGES": config.build.merged_system_pkgs(),
            "PIP_PACKAGES": config.build.merged_python_pkgs(),
            "CUSTOM_CMD": config.build.merged_custom_cmds(),
        }

        print(
            f"Building dockerfile={dockerfile} "
            f"in context={contextdir} "
            f"with args={build_args}"
        )

        try:
            client = docker.from_env()

            # TODO: Push logs to mongo/seaweed?
            client.images.build(
                path=str(contextdir),
                dockerfile=dockerfile,
                pull=True,
                tag=f"{self.CUSTOM_IMAGE_PREFIX}-{type}:{imgtag}",
                timeout=self.BUILD_MAX_WAIT,
                buildargs=build_args,
            )

            return True
        except docker.errors.BuildError as e:
            raise e
        except docker.errors.APIError as e:
            raise e
        except Exception as e:
            raise e

    def find_worker_ctx(self, type: str) -> Path:
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
            Path(self.DOCKERFILE_PROD_PATH, filename).resolve(),
            Path(__file__, self.DOCKERFILE_DEV_PATH, filename).resolve(),
        ]
        for path in lookup_paths:
            if path.exists():
                return path.parent, filename

        raise FileNotFoundError(f"Cannot find the {filename}")
