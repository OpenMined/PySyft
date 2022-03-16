# third party
import boto3
from botocore.client import Config
from pydantic import BaseSettings

# relative
from ...grid.grid_url import GridURL


def get_s3_client(settings: BaseSettings = BaseSettings()) -> "boto3.client.S3":
    try:
        s3_endpoint = settings.S3_ENDPOINT
        s3_port = settings.S3_PORT
        s3_grid_url = GridURL(host_or_ip=s3_endpoint, port=s3_port)
        return boto3.client(
            "s3",
            endpoint_url=s3_grid_url.url,
            aws_access_key_id=settings.S3_ROOT_USER,
            aws_secret_access_key=settings.S3_ROOT_PWD,
            config=Config(signature_version="s3v4"),
            region_name=settings.S3_REGION,
        )
    except Exception as e:
        print(f"Failed to create S3 Client with {s3_endpoint} {s3_port} {s3_grid_url}")
        raise e
