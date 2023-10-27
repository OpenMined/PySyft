# stdlib
import json
import subprocess

# third party
from flask import Flask
from flask import request

# Flask application instance
app = Flask(__name__)


@app.route("/configure_azure", methods=["POST"])
def test() -> str:
    first_res = json.loads(request.data.decode("utf-8").replace("'", '"'))
    account_name = first_res["account_name"]
    account_key = first_res["account_key"]
    container_name = first_res["container_name"]
    remote_name = first_res["remote_name"]
    bucket_name = first_res["bucket_name"]

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
