# stdlib
import os

# third party
import boto3
from botocore.client import Config

# relative
from ...grid.grid_url import GridURL


def get_s3_client(docker_host: bool = False) -> "boto3.client.S3":
    try:
        s3_endpoint = str(os.environ.get("S3_ENDPOINT", "")).strip()
        s3_port = int(os.environ.get("S3_PORT", "8333"))
        s3_grid_url = GridURL(host_or_ip=s3_endpoint, port=s3_port)
        if docker_host:
            s3_grid_url = s3_grid_url.as_docker_host()

        return boto3.client(
            "s3",
            endpoint_url=s3_grid_url.url,
            aws_access_key_id=os.getenv("S3_ROOT_USER", ""),
            aws_secret_access_key=os.getenv("S3_ROOT_PWD", ""),
            config=Config(signature_version="s3v4"),
            region_name=os.getenv("S3_REGION", "us-east-1"),
        )
    except Exception as e:
        print(f"Failed to create S3 Client with {s3_endpoint} {s3_port} {s3_grid_url}")
        raise e
