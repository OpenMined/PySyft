# third party
import boto3
from botocore.client import Config as BotoConfig

# syft absolute
from syft.grid.grid_url import GridURL

# grid absolute
from grid.core.config import settings


def get_s3_client(docker_host: bool = False) -> boto3.client:
    """Returns the s3 boto3 client.

    Args:
        docker_host (bool, optional): If True, then s3 endpoint maps to the docker host. Defaults to False.

    Returns:
        boto3.client: boto3 s3 client.
    """
    if docker_host:
        s3_grid_url = GridURL(
            host_or_ip=settings.S3_ENDPOINT, port=settings.S3_PORT
        ).as_docker_host()
    else:
        s3_grid_url = GridURL(host_or_ip=settings.S3_ENDPOINT, port=settings.S3_PORT)

    # Initialize the boto3 client for s3
    client = boto3.client(
        "s3",
        endpoint_url=s3_grid_url.url,
        aws_access_key_id=settings.S3_ROOT_USER,
        aws_secret_access_key=settings.S3_ROOT_PWD,
        config=BotoConfig(signature_version="s3v4"),
        region_name=settings.S3_REGION,
    )
    return client
