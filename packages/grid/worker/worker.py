# stdlib
from typing import Any

# third party
from fastapi import APIRouter
from fastapi import Body
from fastapi import Depends
from fastapi import FastAPI
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
from syft.core.node.new.user import UserPrivateKey
from syft.core.node.new.user_service import UserService
from syft.core.node.worker import Worker

API_V1_SYFT_STR = "/api/v1/syft/worker"
router = APIRouter()


worker = Worker()
print("worker", worker.id, worker.signing_key)


async def get_body(request: Request) -> bytes:
    return await request.body()


@router.get("/", response_model=str)
def root() -> Response:
    return Response(
        b"hello",
        media_type="application/octet-stream",
    )


@router.get("/new_api", response_model=str)
def syft_new_api() -> Response:
    return Response(
        serialize(worker.get_api(), to_bytes=True),
        media_type="application/octet-stream",
    )


@router.post("/new_api_call", response_model=str)
def syft_new_api_call(request: Request, data: bytes = Depends(get_body)) -> Any:
    obj_msg = deserialize(blob=data, from_bytes=True)
    result = worker.handle_api_call(api_call=obj_msg)
    return Response(
        serialize(result, to_bytes=True),
        media_type="application/octet-stream",
    )


@router.post(
    "/new_login", name="new_login", status_code=200, response_class=JSONResponse
)
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


app = FastAPI(title="Worker")
app.include_router(router, prefix=API_V1_SYFT_STR)
