# future
from __future__ import annotations

# stdlib
import secrets
from typing import Any
from typing import TYPE_CHECKING

# third party
from typing_extensions import Self

# relative
from ...abstract_server import AbstractServer
from ...client.client import HTTPConnection
from ...client.client import PythonConnection
from ...client.client import ServerConnection
from ...client.client import SyftClient
from ...serde.serializable import serializable
from ...server.worker_settings import WorkerSettings
from ...types.errors import SyftException
from ...types.result import as_result
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.transforms import TransformContext
from ...types.uid import UID
from ..context import AuthedServiceContext
from ..context import ServerServiceContext

if TYPE_CHECKING:
    # relative
    from .server_peer import ServerPeer


@serializable(canonical_name="ServerRoute", version=1)
class ServerRoute:
    def client_with_context(self, context: ServerServiceContext) -> SyftClient:
        """
        Convert the current route (self) to a connection (either HTTP, Veilid or Python)
        and create a SyftClient from the connection.

        Args:
            context (ServerServiceContext): The ServerServiceContext containing the server information.

        Returns:
            SyftClient: Returns the created SyftClient
        """
        connection = route_to_connection(route=self, context=context)
        client_type = connection.get_client_type().unwrap()
        return client_type(
            connection=connection, credentials=context.server.signing_key
        )

    @as_result(SyftException)
    def validate_with_context(self, context: AuthedServiceContext) -> ServerPeer:
        # relative
        from .server_peer import ServerPeer

        # Step 1: Check if the given route is able to reach the given server
        # As we allow the user to give custom routes, we need to check the reachability of the route
        self_client = self.client_with_context(context=context)

        # generating a random challenge
        random_challenge = secrets.token_bytes(16)
        challenge_signature = self_client.api.services.network.ping(random_challenge)
        try:
            # Verifying if the challenge is valid
            context.server.verify_key.verify_key.verify(
                random_challenge, challenge_signature
            )
        except Exception:
            raise SyftException(public_message="Signature Verification Failed in ping")

        # Step 2: Create a Server Peer with the given route
        self_server_peer: ServerPeer = context.server.settings.to(ServerPeer)
        self_server_peer.server_routes.append(self)

        return self_server_peer


@serializable()
class HTTPServerRoute(SyftObject, ServerRoute):
    __canonical_name__ = "HTTPServerRoute"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID | None = None  # type: ignore
    host_or_ip: str
    private: bool = False
    protocol: str = "http"
    port: int = 80
    proxy_target_uid: UID | None = None
    priority: int = 1
    rtunnel_token: str | None = None

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, HTTPServerRoute):
            return False
        return hash(self) == hash(other)

    def __hash__(self) -> int:
        return (
            hash(self.host_or_ip)
            + hash(self.port)
            + hash(self.protocol)
            + hash(self.proxy_target_uid)
            + hash(self.rtunnel_token)
        )

    def __str__(self) -> str:
        return f"{self.protocol}://{self.host_or_ip}:{self.port}"


@serializable()
class PythonServerRoute(SyftObject, ServerRoute):
    __canonical_name__ = "PythonServerRoute"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID | None = None  # type: ignore
    worker_settings: WorkerSettings
    proxy_target_uid: UID | None = None
    priority: int = 1

    @property
    def server(self) -> AbstractServer | None:
        # relative
        from ...server.worker import Worker

        server = Worker(
            id=self.worker_settings.id,
            name=self.worker_settings.name,
            server_type=self.worker_settings.server_type,
            server_side_type=self.worker_settings.server_side_type,
            signing_key=self.worker_settings.signing_key,
            db_config=self.worker_settings.db_config,
            processes=1,
        )
        return server

    @classmethod
    def with_server(cls, server: AbstractServer) -> Self:
        worker_settings = WorkerSettings.from_server(server)
        return cls(id=worker_settings.id, worker_settings=worker_settings)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, PythonServerRoute):
            return False
        return hash(self) == hash(other)

    def __hash__(self) -> int:
        return (
            hash(self.worker_settings.id)
            + hash(self.worker_settings.name)
            + hash(self.worker_settings.server_type)
            + hash(self.worker_settings.server_side_type)
            + hash(self.worker_settings.signing_key)
        )

    def __str__(self) -> str:
        return "PythonServerRoute"


@serializable()
class VeilidServerRoute(SyftObject, ServerRoute):
    __canonical_name__ = "VeilidServerRoute"
    __version__ = SYFT_OBJECT_VERSION_1

    vld_key: str
    proxy_target_uid: UID | None = None
    priority: int = 1

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, VeilidServerRoute):
            return False
        return hash(self) == hash(other)

    def __hash__(self) -> int:
        return hash(self.vld_key) + hash(self.proxy_target_uid)


ServerRouteTypeV1 = HTTPServerRoute | PythonServerRoute | VeilidServerRoute
ServerRouteType = HTTPServerRoute | PythonServerRoute


def route_to_connection(
    route: ServerRoute, context: TransformContext | None = None
) -> ServerConnection:
    if isinstance(route, HTTPServerRoute):
        return route.to(HTTPConnection, context=context)
    elif isinstance(route, PythonServerRoute):
        return route.to(PythonConnection, context=context)
    else:
        raise ValueError(f"Route {route} is not supported.")


def connection_to_route(connection: ServerConnection) -> ServerRoute:
    if isinstance(connection, HTTPConnection):
        return connection.to(HTTPServerRoute)
    elif isinstance(connection, PythonConnection):  # type: ignore[unreachable]
        return connection.to(PythonServerRoute)
    else:
        raise ValueError(f"Connection {connection} is not supported.")
