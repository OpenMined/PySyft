# stdlib
from typing import Any

# third party
from fastapi import APIRouter
from fastapi import Body
from fastapi import Depends
from fastapi import Request
from fastapi import Response
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import ValidationError

# syft absolute
from syft import deserialize
from syft import serialize  # type: ignore
from syft.core.node.new.context import UnauthedServiceContext
from syft.core.node.new.credentials import UserLoginCredentials
from syft.core.node.new.node import NewNode
from syft.core.node.new.node_metadata import NodeMetadataJSON
from syft.core.node.new.user import UserPrivateKey
from syft.core.node.new.user_service import UserService
from syft.telemetry import TRACE_MODE

# grid absolute
from grid.core.node import worker

if TRACE_MODE:
    # third party
    from opentelemetry import trace
    from opentelemetry.propagate import extract


router = APIRouter()


async def get_body(request: Request) -> bytes:
    return await request.body()


# provide information about the node in JSON
@router.get("/metadata", response_class=JSONResponse)
async def syft_metadata() -> JSONResponse:
    return worker.metadata().to(NodeMetadataJSON)


def handle_syft_new_api() -> Response:
    return Response(
        serialize(worker.get_api(), to_bytes=True),
        media_type="application/octet-stream",
    )


# get the SyftAPI object
@router.get("/api")
async def syft_new_api(request: Request) -> Response:
    if TRACE_MODE:
        with trace.get_tracer(syft_new_api.__module__).start_as_current_span(
            syft_new_api.__qualname__,
            context=extract(request.headers),
            kind=trace.SpanKind.SERVER,
        ):
            return handle_syft_new_api()
    else:
        return handle_syft_new_api()


def handle_new_api_call(data: bytes) -> Response:
    obj_msg = deserialize(blob=data, from_bytes=True)
    result = worker.handle_api_call(api_call=obj_msg)
    return Response(
        serialize(result, to_bytes=True),
        media_type="application/octet-stream",
    )


# make a request to the SyftAPI
@router.post("/api_call")
async def syft_new_api_call(
    request: Request, data: bytes = Depends(get_body)
) -> Response:
    if TRACE_MODE:
        with trace.get_tracer(syft_new_api_call.__module__).start_as_current_span(
            syft_new_api_call.__qualname__,
            context=extract(request.headers),
            kind=trace.SpanKind.SERVER,
        ):
            return handle_new_api_call(data)
    else:
        return handle_new_api_call(data)


def handle_login(email: str, password: str, node: NewNode) -> Any:
    try:
        login_credentials = UserLoginCredentials(email=email, password=password)
    except ValidationError as e:
        return {"Error": e.json()}

    method = node.get_service_method(UserService.exchange_credentials)
    context = UnauthedServiceContext(node=node, login_credentials=login_credentials)
    result = method(context=context)

    if result.is_err():
        logger.bind(payload={"email": email}).error(result.err())
        response = {"Error": result.err()}
    else:
        user_private_key = result.ok()
        if not isinstance(user_private_key, UserPrivateKey):
            raise Exception(f"Incorrect return type: {type(user_private_key)}")
        response = user_private_key

    return Response(
        serialize(response, to_bytes=True),
        media_type="application/octet-stream",
    )


# exchange email and password for a SyftSigningKey
@router.post("/login", name="login", status_code=200)
async def login(
    request: Request,
    email: str = Body(..., example="info@openmined.org"),
    password: str = Body(..., example="changethis"),
) -> Any:
    if TRACE_MODE:
        with trace.get_tracer(login.__module__).start_as_current_span(
            login.__qualname__,
            context=extract(request.headers),
            kind=trace.SpanKind.SERVER,
        ):
            return handle_login(email, password, worker)
    else:
        return handle_login(email, password, worker)
