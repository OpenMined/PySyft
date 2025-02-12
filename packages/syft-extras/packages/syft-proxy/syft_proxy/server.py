import json
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from loguru import logger
from syft_core import Client
from syft_rpc import rpc, rpc_db
from syft_rpc.protocol import SyftFuture, SyftResponse

from syft_proxy.cli import __version__
from syft_proxy.models import (
    RPCSendRequest,
    RPCStatus,
    RPCStatusCode,
)

HEADER_APP_NAME = "x-app-name"
HEADER_SYFTBOX_URL = "x-syftbox-url"
HEADER_SYFTBOX_URLS = "x-syftbox-urls"
RPC_REQUEST_EXPIRY = "30s"

client = Client.load()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],  # Allows all origins
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


ascii_art = rf"""
 ____         __ _   ____
/ ___| _   _ / _| |_| __ )  _____  __
\___ \| | | | |_| __|  _ \ / _ \ \/ /
 ___) | |_| |  _| |_| |_) | (_) >  <
|____/ \__, |_|  \__|____/ \___/_/\_\
       |___/        {__version__:>17}

SyftBox HTTP Proxy
"""


@app.get("/", response_class=PlainTextResponse)
async def index():
    return ascii_art


@app.get("/info")
async def info():
    return {"version": __version__}


@app.post("/rpc")
async def rpc_send(rpc_req: RPCSendRequest, blocking: bool = False):
    try:
        future: SyftFuture = rpc.send(
            client=client,
            url=rpc_req.url,
            headers=rpc_req.headers,
            body=rpc_req.body,
            expiry=rpc_req.expiry,
        )

        if not blocking:
            logger.info(
                f"Non-blocking RPC request {future.id} sent to {future.request.url}"
            )
            app_name = f"proxy-{rpc_req.app_name}"
            rpc_db.save_future(future, app_name)
            return RPCStatus(
                id=str(future.id),
                status=RPCStatusCode.PENDING,
                request=future.request,
                response=None,
            ).model_dump(mode="json")
        else:
            logger.info(
                f"Blocking RPC request {future.id} sent to {future.request.url}"
            )
            result: SyftResponse = future.wait()
            return JSONResponse(
                status_code=int(result.status_code),
                content=RPCStatus(
                    id=str(result.id),
                    status=RPCStatusCode.COMPLETED,
                    request=future.request,
                    response=result,
                ).model_dump(mode="json"),
            )
    except Exception as ex:
        logger.error(f"Error sending RPC request: {ex}")
        raise HTTPException(status_code=500, detail=str(ex))


@app.get("/rpc/schema/{app_name}")
async def rpc_schema(app_name: str):
    try:
        app_path = client.api_data(app_name)
        app_schema = app_path / "rpc" / "rpc.schema.json"
        return json.loads(app_schema.read_text())
    except Exception as ex:
        logger.error(f"Error sending RPC request: {ex}")
        raise HTTPException(status_code=500, detail=str(ex))


@app.get("/rpc/status/{id}")
async def rpc_status(id: str):
    # try to get future from the db
    try:
        future = rpc_db.get_future(id)
    except Exception as ex:
        logger.error(f"RPC future {id}: EXCEPTION {ex}")
        raise HTTPException(status_code=500, detail=str(ex))

    if not future:
        logger.info(f"RPC future {id}: NOT FOUND")
        return JSONResponse(
            status_code=404,
            content=RPCStatus(
                id=id,
                status=RPCStatusCode.NOT_FOUND,
                request=None,
                response=None,
            ).model_dump(mode="json"),
        )

    logger.info(f"RPC future {id}: FOUND")
    result: Optional[SyftResponse] = future.resolve()

    if result is None:
        logger.info(f"RPC future {id}: PENDING")
        return JSONResponse(
            content=RPCStatus(
                id=id,
                status=RPCStatusCode.PENDING,
                request=future.request,
                response=None,
            ).model_dump(mode="json")
        )
    elif not result.is_success:
        logger.info(f"RPC future {id}: ERROR")
        rpc_db.delete_future(id)
        return JSONResponse(
            status_code=int(result.status_code),
            headers=result.headers,
            content=RPCStatus(
                id=id,
                status=RPCStatusCode.ERROR,
                request=future.request,
                response=result,
            ).model_dump(mode="json"),
        )
    else:
        logger.info(f"RPC future {id}: COMPLETED")
        rpc_db.delete_future(id)
        logger.debug(result.json())
        return JSONResponse(
            status_code=int(result.status_code),
            headers=result.headers,
            content=RPCStatus(
                id=id,
                status=RPCStatusCode.COMPLETED,
                request=future.request,
                response=result,
            ).model_dump(mode="json"),
        )


# @app.post("/llm/chat")
# async def chat(request: Request):
#     return JSONResponse(result)
