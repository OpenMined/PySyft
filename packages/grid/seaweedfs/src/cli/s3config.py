# stdlib
import argparse
import json
from pathlib import Path
from secrets import token_hex


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./s3_config.json")
    parser.add_argument("--username", type=str, default="admin")
    parser.add_argument("--password", type=str, default=None)

    args = parser.parse_args()
    config = Path(args.config)
    username = args.username
    password = args.password or token_hex(16)

    create_s3_config(config, username, password)
