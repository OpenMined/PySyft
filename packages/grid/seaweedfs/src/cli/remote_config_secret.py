# stdlib
import json
from pathlib import Path

# third party
from typer import Typer

# relative
from ..mount_options import BucketType

cli = Typer(
    name="Get remote.configure secret args",
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False,
    pretty_exceptions_short=False,
)


@cli.command()
def get_secret(type: BucketType, secret_path: Path) -> None:
    creds = json.loads(secret_path.read_text())

    if type == BucketType.S3:
        print(
            f"-s3.access_key={creds['aws_access_key_id']} -s3.secret_key={creds['aws_secret_access_key']}"
        )
    elif type == BucketType.AZURE:
        print(
            f"-azure.account_name={creds['azure_account_name']} -azure.account_key={creds['azure_account_key']}"
        )
    elif type == BucketType.GCS:
        print(f"-gcs.appCredentialsFile={secret_path}")


if __name__ == "__main__":
    cli()
