# stdlib
import json
import lzma
import os
import sys
from typing import Annotated

# third party
from fastapi import Body
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from loguru import logger

# relative
from .models import ResponseModel
from .models import TestVeilidStreamerRequest
from .models import TestVeilidStreamerResponse
from .utils import generate_random_alphabets
from .veilid_connection_singleton import VeilidConnectionSingleton
from .veilid_core import app_call
from .veilid_core import app_message
from .veilid_core import generate_vld_key
from .veilid_core import healthcheck
from .veilid_core import retrieve_vld_key

# Logging Configuration
log_level = os.getenv("APP_LOG_LEVEL", "INFO").upper()
logger.remove()
logger.add(sys.stderr, colorize=True, level=log_level)

app = FastAPI(title="Veilid")
veilid_conn = VeilidConnectionSingleton()


@app.get("/", response_model=ResponseModel)
async def read_root() -> ResponseModel:
    return ResponseModel(message="Veilid has started")


@app.get("/healthcheck", response_model=ResponseModel)
async def healthcheck_endpoint() -> ResponseModel:
    res = await healthcheck()
    if res:
        return ResponseModel(message="OK")
    else:
        return ResponseModel(message="FAIL")


@app.post("/generate_vld_key", response_model=ResponseModel)
async def generate_vld_key_endpoint() -> ResponseModel:
    try:
        res = await generate_vld_key()
        return ResponseModel(message=res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate VLD key: {e}")


@app.get("/retrieve_vld_key", response_model=ResponseModel)
async def retrieve_vld_key_endpoint() -> ResponseModel:
    try:
        res = await retrieve_vld_key()
        return ResponseModel(message=res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/app_message", response_model=ResponseModel)
async def app_message_endpoint(
    request: Request, vld_key: Annotated[str, Body()], message: Annotated[bytes, Body()]
) -> ResponseModel:
    try:
        logger.info("Received app_message request")
        res = await app_message(vld_key=vld_key, message=message)
        return ResponseModel(message=res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/app_call")
async def app_call_endpoint(
    request: Request, vld_key: Annotated[str, Body()], message: Annotated[bytes, Body()]
) -> Response:
    try:
        res = await app_call(vld_key=vld_key, message=message)
        return Response(res, media_type="application/octet-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.api_route("/proxy", methods=["GET", "POST", "PUT"])
async def proxy(request: Request) -> Response:
    logger.info("Proxying request")

    request_data = await request.json()
    logger.info(f"Request URL: {request_data}")

    vld_key = request_data.get("vld_key")
    request_data.pop("vld_key")
    message = json.dumps(request_data).encode()

    res = await app_call(vld_key=vld_key, message=message)
    decompressed_res = lzma.decompress(res)
    return Response(decompressed_res, media_type="application/octet-stream")


@app.on_event("startup")
async def startup_event() -> None:
    try:
        await veilid_conn.initialize_connection()
    except Exception as e:
        logger.exception(f"Failed to connect to Veilid: {e}")
        raise e


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await veilid_conn.release_connection()


@app.post("/test_veilid_streamer")
async def test_veilid_streamer(
    request_data: TestVeilidStreamerRequest,
) -> TestVeilidStreamerResponse:
    """Test endpoint for notebooks/Testing/Veilid/Large-Message-Testing.ipynb.

    This endpoint is used to test the Veilid streamer by receiving a request body of any
    arbitrary size and sending back a response of a size specified in the request body.
    The length of the response body is determined by the `expected_response_length` field
    in the request body. After adding the necessary fields, both the request and response
    bodies are padded with random alphabets to reach the expected length using a
    `random_padding` field.
    """
    expected_response_length = request_data.expected_response_length
    if expected_response_length <= 0:
        raise HTTPException(status_code=400, detail="Length must be greater than zero")

    try:
        request_body_length = len(json.dumps(request_data.dict()))
        response = TestVeilidStreamerResponse(
            received_request_body_length=request_body_length,
            random_padding="",
        )
        padding_length = expected_response_length - request_body_length
        random_message = generate_random_alphabets(padding_length)
        response.random_padding = random_message
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
