# third party
from fastapi import FastAPI

# relative
from .veilid_core import VeilidConnectionSingleton
from .veilid_core import generate_dht_key
from .veilid_core import get_veilid_conn
from .veilid_core import retrieve_dht_key

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


@app.post("/generate_dht_key")
async def generate_dht_key_endpoint() -> dict[str, str]:
    return await generate_dht_key()


@app.get("/retrieve_dht_key")
async def retrieve_dht_key_endpoint() -> dict[str, str]:
    return await retrieve_dht_key()


@app.on_event("startup")
async def startup_event() -> None:
    try:
        await veilid_conn.initialize_connection()
    except Exception as e:
        # TODO: Shift to Logging Module
        print(e)
        raise e


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await veilid_conn.release_connection()
