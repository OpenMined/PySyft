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
from syft.core.node.new.node_metadata import NodeMetadataJSON
from syft.core.node.new.user import UserPrivateKey
from syft.core.node.new.user_service import UserService

# grid absolute
from grid.core.node import node
from grid.core.node import worker

router = APIRouter()


async def get_body(request: Request) -> bytes:
    return await request.body()


# provide information about the node in JSON
@router.get("/metadata", response_class=JSONResponse)
def syft_metadata() -> JSONResponse:
    return worker.metadata().to(NodeMetadataJSON)


# get the SyftAPI object
@router.get("/api")
def syft_new_api() -> Response:
    return Response(
        serialize(node.get_api(), to_bytes=True),
        media_type="application/octet-stream",
    )


# make a request to the SyftAPI
@router.post("/api_call")
def syft_new_api_call(request: Request, data: bytes = Depends(get_body)) -> Response:
    obj_msg = deserialize(blob=data, from_bytes=True)
    result = worker.handle_api_call(api_call=obj_msg)
    return Response(
        serialize(result, to_bytes=True),
        media_type="application/octet-stream",
    )


# exchange email and password for a SyftSigningKey
@router.post("/login", name="login", status_code=200)
def login(
    email: str = Body(..., example="info@openmined.org"),
    password: str = Body(..., example="changethis"),
) -> Any:
    try:
        login_credentials = UserLoginCredentials(email=email, password=password)
    except ValidationError as e:
        return {"Error": e.json()}

    method = worker.get_service_method(UserService.exchange_credentials)
    context = UnauthedServiceContext(node=worker, login_credentials=login_credentials)
    result = method(context=context)
    if result.is_err():
        logger.bind(payload={"email": email}).error(result.err())
        return {"Error": result.err()}

    user_private_key = result.ok()
    if not isinstance(user_private_key, UserPrivateKey):
        raise Exception(f"Incorrect return type: {type(user_private_key)}")

    return Response(
        serialize(user_private_key, to_bytes=True),
        media_type="application/octet-stream",
    )
