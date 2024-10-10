# stdlib
import os

# syft absolute
import syft as sy
from syft.orchestra import DeploymentType
from syft.util.test_helpers.worker_helpers import (
    build_and_launch_worker_pool_from_docker_str,
)

# relative
from ..sim.core import SimulatorContext
from .utils import server_info

__all__ = ["bq_create_pool"]


def bq_create_pool(
    ctx: SimulatorContext,
    admin_client: sy.DatasiteClient,
    worker_pool="biquery-pool",
    external_registry_url="k3d-registry.localhost:5800",
):
    base_image = admin_client.images.get_all()[0]
    worker_image_tag = str(base_image.image_identifier).replace(
        "syft-backend", worker_pool
    )

    worker_dockerfile = (
        f"FROM {str(base_image.image_identifier)}\n"
        f"RUN uv pip install db-dtypes google-cloud-bigquery"
    )

    msg = (
        f"Admin {admin_client.metadata.server_side_type}: "
        f"Worker Pool tag '{worker_image_tag}' on {server_info(admin_client)}"
    )

    ctx.logger.info(f"{msg} - Creating")

    deployment_type = os.environ.get("ORCHESTRA_DEPLOYMENT_TYPE", DeploymentType.PYTHON)

    build_and_launch_worker_pool_from_docker_str(
        environment=str(deployment_type),
        client=admin_client,
        worker_pool_name=worker_pool,
        worker_dockerfile=worker_dockerfile,
        external_registry=external_registry_url,
        docker_tag=worker_image_tag,
        custom_pool_pod_annotations=None,
        custom_pool_pod_labels=None,
        scale_to=3,
    )
    ctx.logger.info(f"{msg} - Created")
