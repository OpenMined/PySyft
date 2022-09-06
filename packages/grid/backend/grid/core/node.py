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

# grid absolute
from grid.core.config import Settings
from grid.core.config import settings
from grid.db.session import get_db_engine

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
    except Exception as e:
        print(f"Failed to create bucket. {e}")
        raise e


if settings.NODE_TYPE.lower() == "domain":
    node = Domain(
        "Domain", db_engine=get_db_engine(), settings=settings, document_store=True
    )
    if settings.USE_BLOB_STORAGE:
        create_s3_bucket(bucket_name=node.id.no_dash, settings=settings)

elif settings.NODE_TYPE.lower() == "network":
    node = Network(
        "Network", db_engine=get_db_engine(), settings=settings, document_store=True
    )
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
else:
    raise Exception(
        "Don't know NODE_TYPE "
        + str(settings.NODE_TYPE)
        + ". Please set "
        + "NODE_TYPE to either 'Domain' or 'Network'."
    )

node.loud_print()

if len(node.setup):  # Check if setup was defined previously
    node.name = node.setup.node_name

# Commented out as we have a singleton class for RoleManager.
# # Moving this to get called WITHIN Domain and Network so that they can operate in standalone mode
# if not len(node.roles):  # Check if roles were registered previously
#     seed_db(get_db_session())


def get_client(signing_key: Optional[SigningKey] = None) -> Client:
    return node.get_client(signing_key=signing_key)
