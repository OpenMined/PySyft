# stdlib
import json
from pathlib import Path
from secrets import token_hex

# first party
from src.buckets import AzureBucket
from src.buckets import AzureCreds
from src.buckets import GCSBucket
from src.buckets import GCSCreds
from src.buckets import S3Bucket
from src.buckets import S3Creds
from src.mount_options import MountOptions


def test_mount_options_s3(random_path: Path) -> None:
    # test creds as obj
    creds_obj = {
        "aws_access_key_id": token_hex(8),
        "aws_secret_access_key": token_hex(8),
    }
    opts = MountOptions(
        **{
            "local_bucket": token_hex(8),
            "remote_bucket": {
                "type": "s3",
                "bucket_name": token_hex(8),
                "creds": creds_obj,
            },
        }
    )

    assert isinstance(opts.remote_bucket, S3Bucket)
    assert isinstance(opts.remote_bucket.creds, S3Creds)

    # test creds as path
    random_path.write_text(json.dumps(creds_obj))
    opts = MountOptions(
        **{
            "local_bucket": token_hex(8),
            "remote_bucket": {
                "type": "s3",
                "bucket_name": token_hex(8),
                "creds": str(random_path),
            },
        }
    )

    assert isinstance(opts.remote_bucket, S3Bucket)
    assert isinstance(opts.remote_bucket.creds, S3Creds)
    assert opts.remote_bucket.creds.aws_access_key_id == creds_obj["aws_access_key_id"]


def test_mount_options_gcs(random_path: Path) -> None:
    # test creds as obj
    creds_obj = {
        "type": "service_account",
        "project_id": token_hex(8),
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "universe_domain": "googleapis.com",
    }
    opts = MountOptions(
        **{
            "local_bucket": token_hex(8),
            "remote_bucket": {
                "type": "gcs",
                "bucket_name": token_hex(8),
                "creds": creds_obj,
            },
        }
    )

    assert isinstance(opts.remote_bucket, GCSBucket)
    assert isinstance(opts.remote_bucket.creds, GCSCreds)

    # test creds as path
    random_path.write_text(json.dumps(creds_obj))
    opts = MountOptions(
        **{
            "local_bucket": token_hex(8),
            "remote_bucket": {
                "type": "gcs",
                "bucket_name": token_hex(8),
                "creds": random_path,
            },
        }
    )

    assert isinstance(opts.remote_bucket, GCSBucket)
    assert isinstance(opts.remote_bucket.creds, GCSCreds)


def test_mount_options_azure(random_path: Path) -> None:
    # test creds as obj
    creds_obj = {
        "azure_account_name": token_hex(8),
        "azure_account_key": token_hex(8),
    }
    opts = MountOptions(
        **{
            "local_bucket": token_hex(8),
            "remote_bucket": {
                "type": "azure",
                "container_name": token_hex(8),
                "creds": creds_obj,
            },
        }
    )
    assert isinstance(opts.remote_bucket, AzureBucket)
    assert isinstance(opts.remote_bucket.creds, AzureCreds)

    # test creds as path
    random_path.write_text(json.dumps(creds_obj))
    opts = MountOptions(
        **{
            "local_bucket": token_hex(8),
            "remote_bucket": {
                "type": "azure",
                "container_name": token_hex(8),
                "creds": random_path,
            },
        }
    )
    assert isinstance(opts.remote_bucket, AzureBucket)
    assert isinstance(opts.remote_bucket.creds, AzureCreds)
    assert (
        opts.remote_bucket.creds.azure_account_name == creds_obj["azure_account_name"]
    )
