# stdlib
import logging
from pathlib import Path
import subprocess

# third party
from fastapi import FastAPI
from fastapi import HTTPException

# first party
from src.automount import automount
from src.mount import mount_bucket
from src.mount_options import MountOptions

# Automount all configured buckets
AUTOMOUNT_CONFIG = Path("./automount.yaml")

# keeping it away from /data/ to avoid these being persisted in a volume
PRIVATE_CONF_DIR = Path("/etc/mounts/")
PRIVATE_CONF_DIR.mkdir(mode=0o600, parents=True, exist_ok=True)

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="SeaweedFS Remote Mount API")


@app.on_event("startup")
async def startup_event() -> None:
    automount(automount_conf=AUTOMOUNT_CONFIG, mount_conf_dir=PRIVATE_CONF_DIR)


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
        raise HTTPException(status_code=400, detail=str(e))
    except subprocess.CalledProcessError as e:
        logger.error(
            f"Mount error: code={e.returncode} stdout={e.stdout} stderr={e.stderr}"
        )
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/configure_azure", deprecated=True)
def configure_azure(first_res: dict) -> str:
    account_name = first_res["account_name"]
    account_key = first_res["account_key"]
    container_name = first_res["container_name"]
    remote_name = first_res["remote_name"]
    bucket_name = first_res["bucket_name"]

    logger.info(
        f"Configuring azure bucket name={bucket_name} remote={remote_name} container={container_name}"
    )

    # popen a daemon process
    res = subprocess.run(
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
