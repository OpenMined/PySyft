# stdlib
from typing import Any
from typing import List
from typing import Union
import os

# third part
import boto3
from botocore.client import Config
from ....grid.grid_url import GridURL



def read_chunks(fp, chunk_size=1024**3):
    """Read data in chunks from the file."""
    while True:
        data = fp.read(chunk_size)
        if not data:
            break
        yield data

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


def listify(x: Any) -> List[Any]:
    """turns x into a list.
    If x is a list or tuple, return as list.
    if x is not a list: return [x]
    if x is None: return []

    Args:
        x (Any): some object

    Returns:
        List[Any]: x, as a list
    """
    return list(x) if isinstance(x, (list, tuple)) else ([] if x is None else [x])
