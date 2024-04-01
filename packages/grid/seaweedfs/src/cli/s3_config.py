# stdlib
import json
import os
from pathlib import Path
from secrets import token_hex

S3_CONFIG_PATH = Path(os.getenv("S3_CONFIG_PATH", "/tmp/s3_config.json"))
S3_ROOT_USER = os.getenv("S3_ROOT_USER", "admin")
S3_ROOT_PWD = os.getenv("S3_ROOT_PWD", token_hex(16))


def create_s3_config(config_path: Path, root_user: str, root_password: str) -> None:
    conf = {
        "identities": [
            {
                "name": "iam",
                "credentials": [
                    {
                        "accessKey": root_user,
                        "secretKey": root_password,
                    }
                ],
                "actions": ["Read", "Write", "List", "Tagging", "Admin"],
                "account": None,
            }
        ],
        "accounts": [],
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(conf))
    print(f"Created S3 config at {config_path}")


if __name__ == "__main__":
    create_s3_config(S3_CONFIG_PATH, S3_ROOT_USER, S3_ROOT_PWD)
