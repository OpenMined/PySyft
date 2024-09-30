# stdlib
import os
from urllib.parse import urlparse

# syft absolute
import syft as sy
from syft.orchestra import DeploymentType

# relative
from ..sim.core import SimulatorContext


def server_info(client: sy.DatasiteClient) -> str:
    url = getattr(client.connection, "url", "python")
    return f"{client.name}(url={url}, side={client.metadata.server_side_type})"


def launch_server(ctx: SimulatorContext, server_url: str, server_name: str):
    deployment_type = os.environ.get("ORCHESTRA_DEPLOYMENT_TYPE", DeploymentType.PYTHON)
    ctx.logger.info(f"Deployment type: {deployment_type}")
    if deployment_type == DeploymentType.PYTHON:
        ctx.logger.info(f"Launching python server '{server_name}' at {server_url}")
        parsed_url = urlparse(server_url)
        port = parsed_url.port
        sy.orchestra.launch(name=server_name, reset=True, dev_mode=True, port=port)
