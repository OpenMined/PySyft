from typing import Union
import os


import boto3
from botocore.client import Config
from ...grid.grid_url import GridURL

def get_s3_client(docker_host: bool = False) -> Union[None, boto3.client]:
    if not os.getenv("S3_ENDPOINT"):
        return None

    if docker_host:
        s3_grid_url = GridURL(
            host_or_ip=os.getenv("S3_ENDPOINT"), port=os.getenv("S3_PORT", "8333")
        ).as_docker_host()
    else:
        s3_grid_url = GridURL(host_or_ip=os.getenv("S3_ENDPOINT"), port=os.getenv("S3_PORT", "8333"))

    return boto3.client(
        "s3",
        endpoint_url=s3_grid_url.url,
        aws_access_key_id=os.getenv("S3_ROOT_USER", ""),
        aws_secret_access_key=os.getenv("S3_ROOT_PWD", ""),
        config=Config(signature_version="s3v4"),
        region_name=os.getenv("S3_REGION", "us-east-1"),
    )
