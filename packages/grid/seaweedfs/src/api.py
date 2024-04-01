# stdlib
import subprocess

# third party
from fastapi import FastAPI

# first party
from src.mount import mount_bucket
from src.mount_options import MountOptions

app = FastAPI(title="SeaweedFS Remote Mount API")


@app.get("/")
def ping() -> dict:
    return {"success": True}


@app.post("/mount/")
def mount(opts: MountOptions) -> dict:
    try:
        result = mount_bucket(opts)
    except FileExistsError as e:
        return {
            "success": False,
            "message": str(e),
        }
    except Exception as e:
        return {
            "success": False,
            "message": str(e),
        }

    # update supervisord
    res = subprocess.run(["supervisorctl", "update"])

    # TODO: query supervisord process status to check if the process is running
    return {
        "config_name": result["config_name"],
        "ctl_return": res.returncode,
    }


@app.post("/configure_azure", deprecated=True)
def configure_azure(first_res: dict) -> str:
    account_name = first_res["account_name"]
    account_key = first_res["account_key"]
    container_name = first_res["container_name"]
    remote_name = first_res["remote_name"]
    bucket_name = first_res["bucket_name"]

    res = subprocess.run(
        [
            "bash",
            "scripts/mount_command.sh",
            remote_name,
            account_name,
            bucket_name,
            container_name,
            account_key,
        ]
    )
    return str(res.returncode)
