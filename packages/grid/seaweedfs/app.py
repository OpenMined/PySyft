# stdlib
import json
import subprocess

# third party
from flask import Flask
from flask import request

# Flask application instance
app = Flask(__name__)


@app.route("/configure_azure", methods=["POST"])
def test(
    # account_name: str,
    # account_key: str,
    # container_name: str,
    # remote_name: str,
    # bucket_name: str,
):
    first_res = json.loads(request.data.decode("utf-8").replace("'", '"'))
    print(first_res)
    account_name: str = first_res["account_name"]
    account_key: str = first_res["account_key"]
    container_name: str = first_res["container_name"]
    remote_name: str = first_res["remote_name"]
    bucket_name: str = first_res["bucket_name"]

    res = subprocess.run(
        [
            "bash",
            "mount_command.sh",
            remote_name,
            account_name,
            bucket_name,
            container_name,
            account_key,
        ]
    )
    return str(res.returncode)
