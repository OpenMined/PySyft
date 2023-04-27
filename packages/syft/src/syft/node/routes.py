# stdlib
from typing import Any
from typing import Dict

# third party
from fastapi import APIRouter
from fastapi import Body
from fastapi import Depends
from fastapi import Request
from fastapi import Response
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import ValidationError

# relative
from ..abstract_node import AbstractNode
from ..serde.deserialize import _deserialize as deserialize
from ..serde.serialize import _serialize as serialize
from ..service.context import NodeServiceContext
from ..service.context import UnauthedServiceContext
from ..service.metadata.metadata_service import MetadataService
from ..service.metadata.node_metadata import NodeMetadataJSON
from ..service.response import SyftError
from ..service.user.user import UserCreate
from ..service.user.user import UserPrivateKey
from ..service.user.user_service import UserService
from ..util.telemetry import TRACE_MODE
from .credentials import SyftVerifyKey
from .credentials import UserLoginCredentials
from .worker import Worker


def make_routes(worker: Worker) -> APIRouter:
    if TRACE_MODE:
        # third party
        from opentelemetry import trace
        from opentelemetry.propagate import extract

    router = APIRouter()

    async def get_body(request: Request) -> bytes:
        return await request.body()

    @router.get(
        "/",
        name="healthcheck",
        status_code=200,
        response_class=JSONResponse,
    )
    def root() -> Dict[str, str]:
        """
        Currently, all service backends must satisfy either of the following requirements to
        pass the HTTP health checks sent to it from the GCE loadbalancer: 1. Respond with a
        200 on '/'. The content does not matter. 2. Expose an arbitrary url as a readiness
        probe on the pods backing the Service.
        """
        return {"status": "ok"}

    # provide information about the node in JSON
    @router.get("/metadata", response_class=JSONResponse)
    def syft_metadata() -> JSONResponse:
        return worker.metadata.to(NodeMetadataJSON)

    @router.get("/metadata_capnp")
    def syft_metadata_capnp() -> Response:
        context = NodeServiceContext(node=worker)
        method = worker.get_method_with_context(MetadataService.get, context)
        result = method()
        return Response(
            serialize(result.ok(), to_bytes=True),
            media_type="application/octet-stream",
        )

    def handle_syft_new_api(user_verify_key: SyftVerifyKey) -> Response:
        return Response(
            serialize(worker.get_api(user_verify_key), to_bytes=True),
            media_type="application/octet-stream",
        )

    # get the SyftAPI object
    @router.get("/api")
    def syft_new_api(request: Request, verify_key: str) -> Response:
        user_verify_key: SyftVerifyKey = SyftVerifyKey.from_string(verify_key)
        if TRACE_MODE:
            with trace.get_tracer(syft_new_api.__module__).start_as_current_span(
                syft_new_api.__qualname__,
                context=extract(request.headers),
                kind=trace.SpanKind.SERVER,
            ):
                return handle_syft_new_api(user_verify_key)
        else:
            return handle_syft_new_api(user_verify_key)

    def handle_new_api_call(data: bytes) -> Response:
        obj_msg = deserialize(blob=data, from_bytes=True)
        result = worker.handle_api_call(api_call=obj_msg)
        return Response(
            serialize(result, to_bytes=True),
            media_type="application/octet-stream",
        )

    # make a request to the SyftAPI
    @router.post("/api_call")
    def syft_new_api_call(
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

    def handle_login(email: str, password: str, node: AbstractNode) -> Any:
        try:
            login_credentials = UserLoginCredentials(email=email, password=password)
        except ValidationError as e:
            return {"Error": e.json()}

        method = node.get_service_method(UserService.exchange_credentials)
        context = UnauthedServiceContext(node=node, login_credentials=login_credentials)
        result = method(context=context)

        if isinstance(result, SyftError):
            logger.bind(payload={"email": email}).error(result.message)
            response = {"Error": result.message}
        else:
            user_private_key = result
            if not isinstance(user_private_key, UserPrivateKey):
                raise Exception(f"Incorrect return type: {type(user_private_key)}")
            response = user_private_key

        return Response(
            serialize(response, to_bytes=True),
            media_type="application/octet-stream",
        )

    def handle_register(data: bytes, node: AbstractNode) -> Any:
        user_create = deserialize(data, from_bytes=True)

        if not isinstance(user_create, UserCreate):
            raise Exception(f"Incorrect type received: {user_create}")

        context = NodeServiceContext(node=node)
        method = node.get_method_with_context(UserService.register, context)

        result = method(new_user=user_create)

        if isinstance(result, SyftError):
            logger.bind(payload={"user": user_create}).error(result.message)
            response = SyftError(message=f"User Registration failed: {result.message}")
        else:
            response = result

        return Response(
            serialize(response, to_bytes=True),
            media_type="application/octet-stream",
        )

    # exchange email and password for a SyftSigningKey
    @router.post("/login", name="login", status_code=200)
    def login(
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

    @router.post("/register", name="register", status_code=200)
    def register(request: Request, data: bytes = Depends(get_body)) -> Any:
        if TRACE_MODE:
            with trace.get_tracer(register.__module__).start_as_current_span(
                register.__qualname__,
                context=extract(request.headers),
                kind=trace.SpanKind.SERVER,
            ):
                return handle_register(data, worker)
        else:
            return handle_register(data, worker)

    return router
