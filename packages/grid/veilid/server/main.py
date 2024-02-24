# third party
from fastapi import FastAPI
import veilid

app = FastAPI(title="Veilid")

HOST = "localhost"
PORT = 5959


@app.get("/")
async def read_root() -> dict[str, str]:
    return {"message": "Hello World"}


async def get_veilid_conn() -> veilid.VeilidAPI:
    async def noop_callback(update: veilid.VeilidUpdate) -> None:
        pass

    return await veilid.json_api_connect(HOST, PORT, noop_callback)


@app.get("/healthcheck")
async def healthcheck() -> dict[str, str]:
    async with await get_veilid_conn() as conn:
        state = await conn.get_state()
        if state.network.started:
            return {"message": "OK"}
        else:
            return {"message": "FAIL"}
