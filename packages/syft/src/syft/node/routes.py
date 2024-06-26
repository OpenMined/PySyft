# stdlib
import base64
import binascii
import logging
from typing import Annotated

# third party
from fastapi import APIRouter
from fastapi import Body
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from pydantic import ValidationError
import requests

# relative
from ..abstract_node import AbstractNode
from ..protocol.data_protocol import PROTOCOL_TYPE
from ..serde.deserialize import _deserialize as deserialize
from ..serde.serialize import _serialize as serialize
from ..service.context import NodeServiceContext
from ..service.context import UnauthedServiceContext
from ..service.metadata.node_metadata import NodeMetadataJSON
from ..service.response import SyftError
from ..service.user.user import UserCreate
from ..service.user.user import UserPrivateKey
from ..service.user.user_service import UserService
from ..types.uid import UID
from ..util.telemetry import TRACE_MODE
from .credentials import SyftVerifyKey
from .credentials import UserLoginCredentials
from .worker import Worker

logger = logging.getLogger(__name__)


def make_routes(worker: Worker) -> APIRouter:
    if TRACE_MODE:
        # third party
        try:
            # third party
            from opentelemetry import trace
            from opentelemetry.propagate import extract
        except Exception as e:
            logger.error("Failed to import opentelemetry", exc_info=e)

    router = APIRouter()

    async def get_body(request: Request) -> bytes:
        return await request.body()

    def _blob_url(peer_uid: UID, presigned_url: str) -> str:
        # relative
        from ..service.network.node_peer import route_to_connection

        network_service = worker.get_service("NetworkService")
        peer = network_service.stash.get_by_uid(worker.verify_key, peer_uid).ok()
        peer_node_route = peer.pick_highest_priority_route()
        connection = route_to_connection(route=peer_node_route)
        url = connection.to_blob_route(presigned_url)

        return str(url)

    @router.get("/stream/{peer_uid}/{url_path}/", name="stream")
    async def stream(peer_uid: str, url_path: str) -> StreamingResponse:
        try:
            url_path_parsed = base64.urlsafe_b64decode(url_path.encode()).decode()
        except binascii.Error:
            raise HTTPException(404, "Invalid `url_path`.")

        peer_uid_parsed = UID.from_string(peer_uid)

        url = _blob_url(peer_uid=peer_uid_parsed, presigned_url=url_path_parsed)

        try:
            resp = requests.get(url=url, stream=True)  # nosec
            resp.raise_for_status()
        except requests.RequestException:
            raise HTTPException(404, "Failed to retrieve data from domain.")

        return StreamingResponse(
            resp.iter_content(chunk_size=None), media_type="text/event-stream"
        )

    @router.get(
        "/",
        name="healthcheck",
        status_code=200,
        response_class=JSONResponse,
    )
    def root() -> dict[str, str]:
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
        result = worker.metadata
        return Response(
            serialize(result, to_bytes=True),
            media_type="application/octet-stream",
        )

    def handle_syft_new_api(
        user_verify_key: SyftVerifyKey, communication_protocol: PROTOCOL_TYPE
    ) -> Response:
        return Response(
            serialize(
                worker.get_api(user_verify_key, communication_protocol), to_bytes=True
            ),
            media_type="application/octet-stream",
        )

    # get the SyftAPI object
    @router.get("/api")
    def syft_new_api(
        request: Request, verify_key: str, communication_protocol: PROTOCOL_TYPE
    ) -> Response:
        user_verify_key: SyftVerifyKey = SyftVerifyKey.from_string(verify_key)
        if TRACE_MODE:
            with trace.get_tracer(syft_new_api.__module__).start_as_current_span(
                syft_new_api.__qualname__,
                context=extract(request.headers),
                kind=trace.SpanKind.SERVER,
            ):
                return handle_syft_new_api(user_verify_key, communication_protocol)
        else:
            return handle_syft_new_api(user_verify_key, communication_protocol)

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
        request: Request, data: Annotated[bytes, Depends(get_body)]
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

    def handle_login(email: str, password: str, node: AbstractNode) -> Response:
        try:
            login_credentials = UserLoginCredentials(email=email, password=password)
        except ValidationError as e:
            return {"Error": e.json()}

        method = node.get_service_method(UserService.exchange_credentials)
        context = UnauthedServiceContext(node=node, login_credentials=login_credentials)
        result = method(context=context)

        if isinstance(result, SyftError):
            logger.error(f"Login Error: {result.message}. user={email}")
            response = result
        else:
            user_private_key = result
            if not isinstance(user_private_key, UserPrivateKey):
                raise Exception(f"Incorrect return type: {type(user_private_key)}")
            response = user_private_key

        return Response(
            serialize(response, to_bytes=True),
            media_type="application/octet-stream",
        )

    def handle_register(data: bytes, node: AbstractNode) -> Response:
        user_create = deserialize(data, from_bytes=True)

        if not isinstance(user_create, UserCreate):
            raise Exception(f"Incorrect type received: {user_create}")

        context = NodeServiceContext(node=node)
        method = node.get_method_with_context(UserService.register, context)

        result = method(new_user=user_create)

        if isinstance(result, SyftError):
            logger.error(
                f"Register Error: {result.message}. user={user_create.model_dump()}"
            )
            response = SyftError(message=f"{result.message}")
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
        email: Annotated[str, Body(example="info@openmined.org")],
        password: Annotated[str, Body(example="changethis")],
    ) -> Response:
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
    def register(
        request: Request, data: Annotated[bytes, Depends(get_body)]
    ) -> Response:
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
