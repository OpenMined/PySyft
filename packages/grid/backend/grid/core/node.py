# stdlib
import logging
import time
from typing import Optional

# third party
from nacl.signing import SigningKey

# syft absolute
from syft import Domain  # type: ignore
from syft import Network  # type: ignore
from syft.core.node.common.client import Client
from syft.core.node.common.util import get_s3_client
from syft.core.node.worker import Worker

# grid absolute
from grid.core.config import Settings
from grid.core.config import settings

SEAWEEDFS_MAX_RETRIES = 5


def create_s3_bucket(bucket_name: str, settings: Settings, attempt: int = 0) -> None:
    logging.info("Trying to connect with SeaweedFS ... ")
    s3_client = get_s3_client(settings=settings)

    # Check if the bucket already exists
    try:
        all_buckets = s3_client.list_buckets()
    except Exception:
        if attempt < SEAWEEDFS_MAX_RETRIES:
            time.sleep(1)
            return create_s3_bucket(
                bucket_name=bucket_name, settings=settings, attempt=attempt + 1
            )
        else:
            raise Exception(
                f"Failed to connect to seaweedfs after {SEAWEEDFS_MAX_RETRIES}."
            )

    bucket_exists = (
        any([bucket["Name"] == bucket_name for bucket in all_buckets["Buckets"]])
        if s3_client
        else False
    )

    # If bucket does not exists, then create a new one.
    try:
        if s3_client and not bucket_exists:
            resp = s3_client.create_bucket(Bucket=bucket_name)
            logging.info(f"Bucket Creation response: {resp}")
    except s3_client.meta.client.exceptions.BucketAlreadyExists:
        logging.info(f"Bucket {bucket_name} Already exists.")
        pass
    except Exception as e:
        print(f"Failed to create bucket. {e}")
        raise e


if settings.NODE_TYPE.lower() == "domain":
    node = Domain("Domain", settings=settings, document_store=True)
    worker = Worker(id=node.id, signing_key=node.signing_key)
    if settings.USE_BLOB_STORAGE:
        create_s3_bucket(bucket_name=node.id.no_dash, settings=settings)

elif settings.NODE_TYPE.lower() == "network":
    node = Network("Network", settings=settings, document_store=True)
    worker = Worker(id=node.id, signing_key=node.signing_key)
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
else:
    raise Exception(
        "Don't know NODE_TYPE "
        + str(settings.NODE_TYPE)
        + ". Please set "
        + "NODE_TYPE to either 'Domain' or 'Network'."
    )


root_user = node.users.find_one(search_params={"email": "newinfo@openmined.org"})
worker.root_user = root_user

node.loud_print()

if len(node.setup):  # Check if setup was defined previously
    node.name = node.setup.node_name
    worker.name = node.setup.node_name


def get_client(signing_key: Optional[SigningKey] = None) -> Client:
    return node.get_client(signing_key=signing_key)
