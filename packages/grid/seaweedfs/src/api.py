# stdlib
import logging
import subprocess

# third party
from fastapi import FastAPI
from fastapi import HTTPException

# first party
from src.mount import mount_bucket
from src.mount_options import MountOptions

app = FastAPI(title="SeaweedFS Remote Mount API")

logger = logging.getLogger("uvicorn.error")


@app.get("/")
def ping() -> dict:
    return {"success": True}


@app.post("/mount/")
def mount(opts: MountOptions, overwrite: bool = False) -> dict:
    try:
        result = mount_bucket(opts, overwrite=overwrite)
        path = result["path"]
        logger.info(f'Mount configured at "{path}"')

        return {
            "success": True,
            "name": result["name"],
        }
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except subprocess.CalledProcessError as e:
        logger.error(
            f"Subprocess error: code={e.returncode} stdout={e.stdout} stderr={e.stderr}"
        )
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/configure_azure", deprecated=True)
def configure_azure(first_res: dict) -> str:
    account_name = first_res["account_name"]
    account_key = first_res["account_key"]
    container_name = first_res["container_name"]
    remote_name = first_res["remote_name"]
    bucket_name = first_res["bucket_name"]

    # popen a daemon process
    res = subprocess.Popen(
        [
            "sh",
            "scripts/mount_command.sh",
            remote_name,
            account_name,
            bucket_name,
            container_name,
            account_key,
        ]
    )
    return str(res.returncode)
