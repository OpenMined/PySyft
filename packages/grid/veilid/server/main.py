# third party
from fastapi import FastAPI

# relative
from .veilid_core import VeilidConnectionSingleton
from .veilid_core import get_veilid_conn

app = FastAPI(title="Veilid")
veilid_conn = VeilidConnectionSingleton()


@app.get("/")
async def read_root() -> dict[str, str]:
    return {"message": "Hello World"}


@app.get("/healthcheck")
async def healthcheck() -> dict[str, str]:
    async with await get_veilid_conn() as conn:
        state = await conn.get_state()
        if state.network.started:
            return {"message": "OK"}
        else:
            return {"message": "FAIL"}


@app.on_event("startup")
async def startup_event() -> None:
    try:
        veilid_conn.initialize_connection()
    except Exception as e:
        # TODO: Shift to Logging Module
        print(e)
        raise e


@app.on_event("shutdown")
async def shutdown_event() -> None:
    veilid_conn.release_connection()
