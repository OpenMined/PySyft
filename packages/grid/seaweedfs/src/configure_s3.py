# stdlib
import json
import os
from pathlib import Path

S3_CONFIG_PATH = Path(os.getenv("S3_CONFIG_PATH", "/root/data/s3_config.json"))

if __name__ == "__main__":
    root_user = os.getenv("S3_ROOT_USER", "admin")
    root_password = os.getenv("S3_ROOT_PWD", "admin")
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
    S3_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    S3_CONFIG_PATH.write_text(json.dumps(conf))
    print(f"Created S3 config file at {S3_CONFIG_PATH}")
