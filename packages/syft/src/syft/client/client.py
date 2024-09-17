# future
from __future__ import annotations

# stdlib
import base64
from collections.abc import Callable
from collections.abc import Generator
from collections.abc import Iterable
from enum import Enum
from getpass import getpass
import json
import logging
import traceback
from typing import Any
from typing import TYPE_CHECKING
from typing import cast

# third party
from argon2 import PasswordHasher
from cachetools import TTLCache
from cachetools import cached
from pydantic import field_validator
import requests
from requests import Response
from requests import Session
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry  # type: ignore[import-untyped]
from typing_extensions import Self

# relative
from .. import __version__
from ..abstract_server import AbstractServer
from ..abstract_server import ServerSideType
from ..abstract_server import ServerType
from ..protocol.data_protocol import DataProtocol
from ..protocol.data_protocol import PROTOCOL_TYPE
from ..protocol.data_protocol import get_data_protocol
from ..serde.deserialize import _deserialize
from ..serde.serializable import serializable
from ..serde.serialize import _serialize
from ..server.credentials import SyftSigningKey
from ..server.credentials import SyftVerifyKey
from ..server.credentials import UserLoginCredentials
from ..service.context import ServerServiceContext
from ..service.metadata.server_metadata import ServerMetadata
from ..service.metadata.server_metadata import ServerMetadataJSON
from ..service.response import SyftSuccess
from ..service.user.user import UserCreate
from ..service.user.user import UserPrivateKey
from ..service.user.user import UserView
from ..service.user.user_roles import ServiceRole
from ..service.user.user_service import UserService
from ..types.errors import SyftException
from ..types.result import as_result
from ..types.server_url import ServerURL
from ..types.syft_object import SYFT_OBJECT_VERSION_1
from ..types.uid import UID
from ..util.util import prompt_warning_message
from ..util.util import thread_ident
from ..util.util import verify_tls
from .api import APIModule
from .api import APIRegistry
from .api import SignedSyftAPICall
from .api import SyftAPI
from .api import SyftAPICall
from .api import debox_signed_syftapicall_response
from .api import post_process_result
from .connection import ServerConnection
from .protocol import SyftProtocol

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # relative
    from ..service.network.server_peer import ServerPeer


def upgrade_tls(url: ServerURL, response: Response) -> ServerURL:
    try:
        if response.url.startswith("https://") and url.protocol == "http":
            # we got redirected to https
            https_url = ServerURL.from_url(response.url).with_path("")
            logger.debug(f"ServerURL Upgraded to HTTPS. {https_url}")
            return https_url
    except Exception as e:
        print(f"Failed to upgrade to HTTPS. {e}")
    return url


def forward_message_to_proxy(
    make_call: Callable,
    proxy_target_uid: UID,
    path: str,
    credentials: SyftSigningKey | None = None,
    args: list | None = None,
    kwargs: dict | None = None,
) -> Any:
    kwargs = {} if kwargs is None else kwargs
    args = [] if args is None else args
    call = SyftAPICall(
        server_uid=proxy_target_uid,
        path=path,
        args=args,
        kwargs=kwargs,
        blocking=True,
    )

    if credentials is None:
        # generate a random signing key
        credentials = SyftSigningKey.generate()

    signed_message: SignedSyftAPICall = call.sign(credentials=credentials)
    signed_result = make_call(signed_message)
    response = debox_signed_syftapicall_response(signed_result).unwrap()
    result = post_process_result(response, unwrap_on_success=True)

    return result


API_PATH = "/api/v2"
DEFAULT_SYFT_UI_PORT = 80
DEFAULT_SYFT_UI_ADDRESS = f"http://localhost:{DEFAULT_SYFT_UI_PORT}"
INTERNAL_PROXY_TO_RATHOLE = "http://proxy:80/rtunnel/"


class Routes(Enum):
    ROUTE_METADATA = f"{API_PATH}/metadata"
    ROUTE_API = f"{API_PATH}/api"
    ROUTE_LOGIN = f"{API_PATH}/login"
    ROUTE_REGISTER = f"{API_PATH}/register"
    ROUTE_API_CALL = f"{API_PATH}/api_call"
    ROUTE_BLOB_STORE = "/blob"
    ROUTE_FORGOT_PASSWORD = f"{API_PATH}/forgot_password"
    ROUTE_RESET_PASSWORD = f"{API_PATH}/reset_password"
    STREAM = f"{API_PATH}/stream"


@serializable(attrs=["proxy_target_uid", "url", "rtunnel_token"])
class HTTPConnection(ServerConnection):
    __canonical_name__ = "HTTPConnection"
    __version__ = SYFT_OBJECT_VERSION_1

    url: ServerURL
    proxy_target_uid: UID | None = None
    routes: type[Routes] = Routes
    session_cache: Session | None = None
    headers: dict[str, str] | None = None
    rtunnel_token: str | None = None

    @field_validator("url", mode="before")
    @classmethod
    def make_url(cls, v: Any) -> Any:
        return (
            ServerURL.from_url(v).as_container_host()
            if isinstance(v, str | ServerURL)
            else v
        )

    def set_headers(self, headers: dict[str, str]) -> None:
        self.headers = headers

    def with_proxy(self, proxy_target_uid: UID) -> Self:
        return HTTPConnection(
            url=self.url,
            proxy_target_uid=proxy_target_uid,
            rtunnel_token=self.rtunnel_token,
        )

    def stream_via(self, proxy_uid: UID, url_path: str) -> ServerURL:
        # Update the presigned url path to
        # <gatewayurl>/<peer_uid>/<presigned_url>
        # url_path_bytes = _serialize(url_path, to_bytes=True)

        url_path_str = base64.urlsafe_b64encode(url_path.encode()).decode()
        stream_url_path = f"{self.routes.STREAM.value}/{proxy_uid}/{url_path_str}/"
        return self.url.with_path(stream_url_path)

    def get_cache_key(self) -> str:
        return str(self.url)

    @property
    def api_url(self) -> ServerURL:
        return self.url.with_path(self.routes.ROUTE_API_CALL.value)

    def to_blob_route(self, path: str, **kwargs: Any) -> ServerURL:
        _path = self.routes.ROUTE_BLOB_STORE.value + path
        return self.url.with_path(_path)

    @property
    def session(self) -> Session:
        if self.session_cache is None:
            session = requests.Session()
            retry = Retry(total=3, backoff_factor=0.5)
            adapter = HTTPAdapter(max_retries=retry)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            self.session_cache = session
        return self.session_cache

    def _make_get(
        self, path: str, params: dict | None = None, stream: bool = False
    ) -> bytes | Iterable:
        if params is None:
            return self._make_get_no_params(path, stream=stream)

        url = self.url

        if self.rtunnel_token:
            self.headers = {} if self.headers is None else self.headers
            url = ServerURL.from_url(INTERNAL_PROXY_TO_RATHOLE)
            self.headers["Host"] = self.url.host_or_ip

        url = url.with_path(path)

        response = self.session.get(
            str(url),
            headers=self.headers,
            verify=verify_tls(),
            proxies={},
            params=params,
            stream=stream,
        )
        if response.status_code != 200:
            raise requests.ConnectionError(
                f"Failed to fetch {url}. Response returned with code {response.status_code}"
            )

        # upgrade to tls if available
        self.url = upgrade_tls(self.url, response)

        return response.content

    @cached(cache=TTLCache(maxsize=128, ttl=300))
    def _make_get_no_params(self, path: str, stream: bool = False) -> bytes | Iterable:
        url = self.url

        if self.rtunnel_token:
            self.headers = {} if self.headers is None else self.headers
            url = ServerURL.from_url(INTERNAL_PROXY_TO_RATHOLE)
            self.headers["Host"] = self.url.host_or_ip

        url = url.with_path(path)

        response = self.session.get(
            str(url),
            headers=self.headers,
            verify=verify_tls(),
            proxies={},
            stream=stream,
        )
        if response.status_code != 200:
            raise requests.ConnectionError(
                f"Failed to fetch {url}. Response returned with code {response.status_code}"
            )

        # upgrade to tls if available
        self.url = upgrade_tls(self.url, response)

        if stream:
            return response.iter_content(chunk_size=None)

        return response.content

    def _make_put(
        self, path: str, data: bytes | Generator, stream: bool = False
    ) -> Response:
        url = self.url

        if self.rtunnel_token:
            url = ServerURL.from_url(INTERNAL_PROXY_TO_RATHOLE)
            self.headers = {} if self.headers is None else self.headers
            self.headers["Host"] = self.url.host_or_ip

        url = url.with_path(path)
        response = self.session.put(
            str(url),
            verify=verify_tls(),
            proxies={},
            data=data,
            headers=self.headers,
            stream=stream,
        )
        if response.status_code != 200:
            raise requests.ConnectionError(
                f"Failed to fetch {url}. Response returned with code {response.status_code}"
            )

        # upgrade to tls if available
        self.url = upgrade_tls(self.url, response)

        return response

    def _make_post(
        self,
        path: str,
        json: dict[str, Any] | None = None,
        data: bytes | None = None,
    ) -> bytes:
        url = self.url

        if self.rtunnel_token:
            url = ServerURL.from_url(INTERNAL_PROXY_TO_RATHOLE)
            self.headers = {} if self.headers is None else self.headers
            self.headers["Host"] = self.url.host_or_ip

        url = url.with_path(path)
        response = self.session.post(
            str(url),
            headers=self.headers,
            verify=verify_tls(),
            json=json,
            proxies={},
            data=data,
        )
        if response.status_code != 200:
            raise requests.ConnectionError(
                f"Failed to fetch {url}. Response returned with code {response.status_code}"
            )

        # upgrade to tls if available
        self.url = upgrade_tls(self.url, response)

        return response.content

    def stream_data(self, credentials: SyftSigningKey) -> Response:
        url = self.url.with_path(self.routes.STREAM.value)
        response = self.session.get(
            str(url), verify=verify_tls(), proxies={}, stream=True, headers=self.headers
        )
        return response

    def get_server_metadata(self, credentials: SyftSigningKey) -> ServerMetadataJSON:
        if self.proxy_target_uid:
            response = forward_message_to_proxy(
                make_call=self.make_call,
                proxy_target_uid=self.proxy_target_uid,
                path="metadata",
                credentials=credentials,
            )
            return response
        else:
            response = self._make_get(self.routes.ROUTE_METADATA.value)
            metadata_json = json.loads(response)
            return ServerMetadataJSON(**metadata_json)

    def get_api(  # type: ignore [override]
        self,
        credentials: SyftSigningKey,
        communication_protocol: int,
        metadata: ServerMetadataJSON | None = None,
    ) -> SyftAPI:
        params = {
            "verify_key": str(credentials.verify_key),
            "communication_protocol": communication_protocol,
        }
        if self.proxy_target_uid:
            obj = forward_message_to_proxy(
                self.make_call,
                proxy_target_uid=self.proxy_target_uid,
                path="api",
                kwargs={
                    "credentials": credentials,
                    "communication_protocol": communication_protocol,
                },
                credentials=credentials,
            )
        else:
            content = self._make_get(self.routes.ROUTE_API.value, params=params)
            obj = _deserialize(content, from_bytes=True)
        obj.connection = self
        obj.signing_key = credentials
        obj.communication_protocol = communication_protocol
        obj.metadata = metadata
        if self.proxy_target_uid:
            obj.server_uid = self.proxy_target_uid
        return cast(SyftAPI, obj)

    def login(
        self,
        email: str,
        password: str,
    ) -> SyftSigningKey | None:
        credentials = {"email": email, "password": password}
        if self.proxy_target_uid:
            response = forward_message_to_proxy(
                self.make_call,
                proxy_target_uid=self.proxy_target_uid,
                path="login",
                kwargs=credentials,
            )
        else:
            response = self._make_post(self.routes.ROUTE_LOGIN.value, credentials)
            response = _deserialize(response, from_bytes=True)
            response = post_process_result(response, unwrap_on_success=True)

        return response

    def forgot_password(
        self,
        email: str,
    ) -> SyftSigningKey | None:
        credentials = {"email": email}
        if self.proxy_target_uid:
            obj = forward_message_to_proxy(
                self.make_call,
                proxy_target_uid=self.proxy_target_uid,
                path="forgot_password",
                kwargs=credentials,
            )
        else:
            response = self._make_post(
                self.routes.ROUTE_FORGOT_PASSWORD.value, credentials
            )
            obj = _deserialize(response, from_bytes=True)

        return obj

    def reset_password(
        self,
        token: str,
        new_password: str,
    ) -> SyftSigningKey | None:
        payload = {"token": token, "new_password": new_password}
        if self.proxy_target_uid:
            obj = forward_message_to_proxy(
                self.make_call,
                proxy_target_uid=self.proxy_target_uid,
                path="reset_password",
                kwargs=payload,
            )
        else:
            response = self._make_post(self.routes.ROUTE_RESET_PASSWORD.value, payload)
            obj = _deserialize(response, from_bytes=True)

        return obj

    def register(self, new_user: UserCreate) -> SyftSigningKey:
        data = _serialize(new_user, to_bytes=True)
        if self.proxy_target_uid:
            response = forward_message_to_proxy(
                self.make_call,
                proxy_target_uid=self.proxy_target_uid,
                path="register",
                kwargs={"new_user": new_user},
            )
        else:
            response = self._make_post(self.routes.ROUTE_REGISTER.value, data=data)
            response = _deserialize(response, from_bytes=True)
            response = post_process_result(response, unwrap_on_success=False)
        return response

    def make_call(self, signed_call: SignedSyftAPICall) -> Any:
        msg_bytes: bytes = _serialize(obj=signed_call, to_bytes=True)

        if self.rtunnel_token:
            api_url = ServerURL.from_url(INTERNAL_PROXY_TO_RATHOLE)
            api_url = api_url.with_path(self.routes.ROUTE_API_CALL.value)
            self.headers = {} if self.headers is None else self.headers
            self.headers["Host"] = self.url.host_or_ip
        else:
            api_url = self.api_url

        response = requests.post(  # nosec
            url=api_url,
            data=msg_bytes,
            headers=self.headers,
        )

        if response.status_code != 200:
            raise requests.ConnectionError(
                f"Failed to fetch metadata. Response returned with code {response.status_code}"
            )

        result = _deserialize(response.content, from_bytes=True)
        return result

    def __repr__(self) -> str:
        return f"{type(self).__name__}: {self.url}"

    def __str__(self) -> str:
        return f"{type(self).__name__}: {self.url}"

    def __hash__(self) -> int:
        return hash(self.proxy_target_uid) + hash(self.url)

    @as_result(SyftException)
    def get_client_type(self) -> type[SyftClient]:
        # TODO: Rasswanth, should remove passing in credentials
        # when metadata are proxy forwarded in the server routes
        # in the gateway fixes PR
        # relative
        from .datasite_client import DatasiteClient
        from .enclave_client import EnclaveClient
        from .gateway_client import GatewayClient

        metadata = self.get_server_metadata(credentials=SyftSigningKey.generate())
        if metadata.server_type == ServerType.DATASITE.value:
            return DatasiteClient
        elif metadata.server_type == ServerType.GATEWAY.value:
            return GatewayClient
        elif metadata.server_type == ServerType.ENCLAVE.value:
            return EnclaveClient
        else:
            raise SyftException(
                public_message=f"Unknown server type {metadata.server_type}"
            )


@serializable()
class PythonConnection(ServerConnection):
    __canonical_name__ = "PythonConnection"
    __version__ = SYFT_OBJECT_VERSION_1

    server: AbstractServer
    proxy_target_uid: UID | None = None

    def with_proxy(self, proxy_target_uid: UID) -> Self:
        return PythonConnection(server=self.server, proxy_target_uid=proxy_target_uid)

    def get_server_metadata(self, credentials: SyftSigningKey) -> ServerMetadataJSON:
        if self.proxy_target_uid:
            response = forward_message_to_proxy(
                make_call=self.make_call,
                proxy_target_uid=self.proxy_target_uid,
                path="metadata",
                credentials=credentials,
            )
            return response
        else:
            return self.server.metadata.to(ServerMetadataJSON)

    def to_blob_route(self, path: str, host: str | None = None) -> ServerURL:
        # TODO: FIX!
        if host is not None:
            return ServerURL(host_or_ip=host, port=8333).with_path(path)
        else:
            return ServerURL(port=8333).with_path(path)

    def get_api(  # type: ignore [override]
        self,
        credentials: SyftSigningKey,
        communication_protocol: int,
        metadata: ServerMetadataJSON | None = None,
    ) -> SyftAPI:
        # todo: its a bit odd to identify a user by its verify key maybe?
        if self.proxy_target_uid:
            obj = forward_message_to_proxy(
                self.make_call,
                proxy_target_uid=self.proxy_target_uid,
                path="api",
                kwargs={
                    "credentials": credentials,
                    "communication_protocol": communication_protocol,
                },
                credentials=credentials,
            )
        else:
            obj = self.server.get_api(
                for_user=credentials.verify_key,
                communication_protocol=communication_protocol,
            )
        obj.connection = self
        obj.signing_key = credentials
        obj.communication_protocol = communication_protocol
        obj.metadata = metadata
        if self.proxy_target_uid:
            obj.server_uid = self.proxy_target_uid
        return obj

    def get_cache_key(self) -> str:
        return str(self.server.id)

    def exchange_credentials(self, email: str, password: str) -> SyftSuccess | None:
        context = self.server.get_unauthed_context(
            login_credentials=UserLoginCredentials(email=email, password=password)
        )
        method = self.server.get_method_with_context(
            UserService.exchange_credentials, context
        )
        try:
            result = method()
        except SyftException:
            raise
        except Exception:
            raise SyftException(
                public_message=f"Exception calling exchange credentials. {traceback.format_exc()}"
            )
        return result

    def login(
        self,
        email: str,
        password: str,
    ) -> SyftSigningKey | None:
        if self.proxy_target_uid:
            result = forward_message_to_proxy(
                self.make_call,
                proxy_target_uid=self.proxy_target_uid,
                path="login",
                kwargs={"email": email, "password": password},
            )

        else:
            result = self.exchange_credentials(email=email, password=password)
            result = post_process_result(result, unwrap_on_success=True)
        return result

    def forgot_password(
        self,
        email: str,
    ) -> SyftSigningKey | None:
        credentials = {"email": email}
        if self.proxy_target_uid:
            obj = forward_message_to_proxy(
                self.make_call,
                proxy_target_uid=self.proxy_target_uid,
                path="forgot_password",
                kwargs=credentials,
            )
        else:
            response = self.server.services.user.forgot_password(
                context=ServerServiceContext(server=self.server), email=email
            )
            obj = post_process_result(response, unwrap_on_success=True)

        return obj

    def reset_password(
        self,
        token: str,
        new_password: str,
    ) -> SyftSigningKey | None:
        payload = {"token": token, "new_password": new_password}
        if self.proxy_target_uid:
            obj = forward_message_to_proxy(
                self.make_call,
                proxy_target_uid=self.proxy_target_uid,
                path="reset_password",
                kwargs=payload,
            )
        else:
            response = self.server.services.user.reset_password(
                context=ServerServiceContext(server=self.server),
                token=token,
                new_password=new_password,
            )
            obj = post_process_result(response, unwrap_on_success=True)

        return obj

    def register(self, new_user: UserCreate) -> SyftSigningKey | None:
        if self.proxy_target_uid:
            response = forward_message_to_proxy(
                self.make_call,
                proxy_target_uid=self.proxy_target_uid,
                path="register",
                kwargs={"new_user": new_user},
            )
        else:
            service_context = ServerServiceContext(server=self.server)
            response = self.server.services.user.register(
                context=service_context, new_user=new_user
            )
            response = post_process_result(response, unwrap_on_success=False)
        return response

    def make_call(self, signed_call: SignedSyftAPICall) -> Any:
        return self.server.handle_api_call(signed_call)

    def __repr__(self) -> str:
        return f"{type(self).__name__}"

    def __str__(self) -> str:
        return f"{type(self).__name__}"

    @as_result(SyftException)
    def get_client_type(self) -> type[SyftClient]:
        # relative
        from .datasite_client import DatasiteClient
        from .enclave_client import EnclaveClient
        from .gateway_client import GatewayClient

        metadata = self.get_server_metadata(credentials=SyftSigningKey.generate())
        if metadata.server_type == ServerType.DATASITE.value:
            return DatasiteClient
        elif metadata.server_type == ServerType.GATEWAY.value:
            return GatewayClient
        elif metadata.server_type == ServerType.ENCLAVE.value:
            return EnclaveClient
        else:
            raise SyftException(message=f"Unknown server type {metadata.server_type}")


@serializable(canonical_name="SyftClient", version=1)
class SyftClient:
    connection: ServerConnection
    metadata: ServerMetadataJSON | None
    credentials: SyftSigningKey | None
    __logged_in_user: str = ""
    __logged_in_username: str = ""
    __user_role: ServiceRole = ServiceRole.NONE

    # informs getattr does not have nasty side effects
    __syft_allow_autocomplete__ = [
        "api",
        "code",
        "jobs",
        "users",
        "settings",
        "notifications",
        "custom_api",
    ]

    def __init__(
        self,
        connection: ServerConnection,
        metadata: ServerMetadataJSON | None = None,
        credentials: SyftSigningKey | None = None,
        api: SyftAPI | None = None,
    ) -> None:
        self.connection = connection
        self.metadata = metadata
        self.credentials: SyftSigningKey | None = credentials
        self._api = api
        self.services: APIModule | None = None
        self.communication_protocol: int | str | None = None
        self.current_protocol: int | str | None = None

        self.post_init()

    def get_env(self) -> str:
        return self.api.services.metadata.get_env()

    def post_init(self) -> None:
        if self.metadata is None:
            self._fetch_server_metadata(self.credentials)
        self.metadata = cast(ServerMetadataJSON, self.metadata)
        self.communication_protocol = self._get_communication_protocol(
            self.metadata.supported_protocols
        )

    def set_headers(self, headers: dict[str, str]) -> None:
        if isinstance(self.connection, HTTPConnection):
            self.connection.set_headers(headers)
            return None
        raise SyftException(  # type: ignore
            public_message="Incompatible connection type."
            + f"Expected HTTPConnection, got {type(self.connection)}"
        )

    def _get_communication_protocol(
        self, protocols_supported_by_server: list
    ) -> int | str:
        data_protocol: DataProtocol = get_data_protocol()
        protocols_supported_by_client: list[PROTOCOL_TYPE] = (
            data_protocol.supported_protocols
        )

        self.current_protocol = data_protocol.latest_version
        common_protocols = set(protocols_supported_by_client).intersection(
            protocols_supported_by_server
        )

        if len(common_protocols) == 0:
            raise Exception(
                "No common communication protocol found between the client and the server."
            )

        if "dev" in common_protocols:
            return "dev"
        return max(common_protocols)

    def create_project(
        self, name: str, description: str, user_email_address: str
    ) -> Any:
        # relative
        from ..service.project.project import ProjectSubmit

        project_create = ProjectSubmit(
            name=name,
            description=description,
            shareholders=[self],
            user_email_address=user_email_address,
            members=[self],
        )
        project = project_create.send()
        return project

    @property
    def authed(self) -> bool:
        return bool(self.credentials)

    @property
    def logged_in_user(self) -> str | None:
        return self.__logged_in_user

    @property
    def logged_in_username(self) -> str | None:
        return self.__logged_in_username

    @property
    def user_role(self) -> ServiceRole:
        return self.__user_role

    @property
    def verify_key(self) -> SyftVerifyKey:
        if self.credentials is None:
            raise ValueError("SigningKey not set on client")
        return self.credentials.verify_key

    @classmethod
    def from_url(cls, url: str | ServerURL) -> Self:
        return cls(connection=HTTPConnection(url=ServerURL.from_url(url)))

    @classmethod
    def from_server(cls, server: AbstractServer) -> Self:
        return cls(connection=PythonConnection(server=server))

    @property
    def name(self) -> str | None:
        return self.metadata.name if self.metadata else None

    @property
    def id(self) -> UID | None:
        return UID.from_string(self.metadata.id) if self.metadata else None

    @property
    def icon(self) -> str:
        return "📡"

    @property
    def peer(self) -> Any:
        # relative
        from ..service.network.network_service import ServerPeer

        return ServerPeer.from_client(self)

    @property
    def route(self) -> Any:
        return self.connection.route

    @property
    def api(self) -> SyftAPI:
        # invalidate API
        if self._api is None or (self._api.signing_key != self.credentials):
            self._fetch_api(self.credentials)
        return cast(SyftAPI, self._api)  # we are sure self._api is not None after fetch

    def guest(self) -> Self:
        return self.__class__(
            connection=self.connection,
            credentials=SyftSigningKey.generate(),
            metadata=self.metadata,
        )

    def exchange_route(
        self,
        client: Self,
        protocol: SyftProtocol = SyftProtocol.HTTP,
        reverse_tunnel: bool = False,
    ) -> SyftSuccess:
        # relative
        from ..service.network.routes import connection_to_route

        if protocol == SyftProtocol.HTTP:
            self_server_route = connection_to_route(self.connection)
            remote_server_route = connection_to_route(client.connection)
            if client.metadata is None:
                raise SyftException(
                    public_message=f"client {client}'s metadata is None!"
                )

            return self.api.services.network.exchange_credentials_with(
                self_server_route=self_server_route,
                remote_server_route=remote_server_route,
                remote_server_verify_key=client.metadata.to(ServerMetadata).verify_key,
                reverse_tunnel=reverse_tunnel,
            )
        else:
            raise ValueError(
                f"Invalid Route Exchange SyftProtocol: {protocol}.Supported protocols are {SyftProtocol.all()}"
            )

    @property
    def jobs(self) -> APIModule | None:
        if self.api.has_service("job"):
            return self.api.services.job
        return None

    @property
    def users(self) -> APIModule | None:
        if self.api.has_service("user"):
            return self.api.services.user
        return None

    @property
    def custom_api(self) -> APIModule | None:
        if self.api.has_service("api"):
            return self.api.services.api
        return None

    @property
    def numpy(self) -> APIModule | None:
        if self.api.has_lib("numpy"):
            return self.api.lib.numpy
        return None

    @property
    def settings(self) -> APIModule | None:
        if self.api.has_service("settings"):
            return self.api.services.settings
        return None

    @property
    def notifications(self) -> APIModule | None:
        if self.api.has_service("notifications"):
            return self.api.services.notifications
        return None

    @property
    def notifier(self) -> APIModule | None:
        if self.api.has_service("notifier"):
            return self.api.services.notifier
        return None

    @property
    def peers(self) -> list[ServerPeer] | None:
        if self.api.has_service("network"):
            return self.api.services.network.get_all_peers()
        return None

    @property
    def account(self) -> UserView | None:
        if self.api.has_service("user"):
            return self.api.services.user.get_current_user()
        return None

    def login_as_guest(self) -> Self:
        _guest_client = self.guest()

        if self.metadata is not None:
            print(
                f"Logged into <{self.name}: {self.metadata.server_side_type.capitalize()}-side "
                f"{self.metadata.server_type.capitalize()}> as GUEST"
            )

        return _guest_client

    # is this used??
    def login_as(self, email: str) -> Self:
        user_private_key = self.api.services.user.key_for_email(email=email)
        if not isinstance(user_private_key, UserPrivateKey):
            return user_private_key
        if self.metadata is not None:
            print(
                f"Logged into <{self.name}: {self.metadata.server_side_type.capitalize()}-side "
                f"{self.metadata.server_type.capitalize()}> as {email}"
            )

        return self.__class__(
            connection=self.connection,
            credentials=user_private_key.signing_key,
            metadata=self.metadata,
        )

    def login(
        self,
        email: str | None = None,
        password: str | None = None,
        cache: bool = True,
        register: bool = False,
        **kwargs: Any,
    ) -> Self:
        if email is None:
            email = input("Email: ")
        if password is None:
            password = getpass("Password: ")

        if register:
            self.register(
                email=email, password=password, password_verify=password, **kwargs
            )

        try:
            user_private_key = self.connection.login(email=email, password=password)
        except Exception as e:
            raise SyftException(public_message=e.public_message)

        signing_key = None if user_private_key is None else user_private_key.signing_key

        client = self.__class__(
            connection=self.connection,
            metadata=self.metadata,
            credentials=signing_key,
        )

        client.__logged_in_user = email

        if user_private_key is not None and client.users is not None:
            client.__user_role = user_private_key.role
            client.__logged_in_username = client.users.get_current_user().name

        if signing_key is not None and client.metadata is not None:
            print(
                f"Logged into <{client.name}: {client.metadata.server_side_type.capitalize()} side "
                f"{client.metadata.server_type.capitalize()}> as <{email}>"
            )
            # relative
            from ..server.server import get_default_root_password

            if password == get_default_root_password():
                message = (
                    "You are using a default password. Please change the password "
                    "using `[your_client].account.set_password([new_password])`."
                )
                prompt_warning_message(message)

            if cache:
                SyftClientSessionCache.add_client(
                    email=email,
                    password=password,
                    connection=client.connection,
                    syft_client=client,
                )
                # Adding another cache storage
                # as this would be useful in retrieving unique clients
                # server uid and verify key are not individually unique
                # both the combination of server uid and verify key are unique
                # which could be used to identity a client uniquely of any given server
                # TODO: It would be better to have a single cache storage
                # combining both email, password and verify key and uid
                SyftClientSessionCache.add_client_by_uid_and_verify_key(
                    verify_key=signing_key.verify_key,
                    server_uid=client.id,
                    syft_client=client,
                )

        # relative
        from ..server.server import CODE_RELOADER

        thread_id = thread_ident()
        if thread_id is not None:
            CODE_RELOADER[thread_id] = client._reload_user_code

        return client

    def _reload_user_code(self) -> None:
        # relative
        from ..service.code.user_code import load_approved_policy_code

        user_code_items = self.code.get_all_for_user()
        load_approved_policy_code(user_code_items=user_code_items, context=None)

    def register(
        self,
        name: str,
        email: str | None = None,
        password: str | None = None,
        password_verify: str | None = None,
        institution: str | None = None,
        website: str | None = None,
    ) -> SyftSigningKey | None:
        if not email:
            email = input("Email: ")
        if not password:
            password = getpass("Password: ")
        if not password_verify:
            password_verify = getpass("Confirm Password: ")
        if password != password_verify:
            raise SyftException(public_message="Passwords do not match")

        try:
            new_user = UserCreate(
                name=name,
                email=email,
                password=password,
                password_verify=password_verify,
                institution=institution,
                website=website,
                created_by=(
                    None if self.__user_role == ServiceRole.GUEST else self.credentials
                ),
            )
        except Exception as e:
            raise SyftException(public_message=str(e))

        if (
            self.metadata
            and self.metadata.server_side_type == ServerSideType.HIGH_SIDE.value
        ):
            message = (
                "You're registering a user to a high side "
                f"{self.metadata.server_type}, which could "
                "host datasets with private information."
            )
            if self.metadata.show_warnings and not prompt_warning_message(
                message=message
            ):
                return None

        return self.connection.register(new_user=new_user)

    def __hash__(self) -> int:
        return hash(self.id) + hash(self.connection)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SyftClient):
            return False
        return (
            self.metadata == other.metadata
            and self.connection == other.connection
            and self.credentials == other.credentials
        )

    def __repr__(self) -> str:
        proxy_target_uid = None
        if self.connection and self.connection.proxy_target_uid:
            proxy_target_uid = self.connection.proxy_target_uid
        client_type = type(self).__name__
        uid = self.id
        if proxy_target_uid:
            client_type = "ProxyClient"
            uid = proxy_target_uid
            return f"<{client_type} - <{uid}>: via {self.connection}>"
        return f"<{client_type} - {self.name} <{uid}>: {self.connection}>"

    def _fetch_server_metadata(self, credentials: SyftSigningKey) -> None:
        metadata = self.connection.get_server_metadata(credentials=credentials)
        if isinstance(metadata, ServerMetadataJSON):
            metadata.check_version(__version__)
            self.metadata = metadata

    def _fetch_api(self, credentials: SyftSigningKey) -> SyftAPI:
        _api: SyftAPI = self.connection.get_api(  # type: ignore [call-arg]
            credentials=credentials,
            communication_protocol=self.communication_protocol,
            metadata=self.metadata,
        )
        self._fetch_server_metadata(self.credentials)

        def refresh_callback() -> SyftAPI:
            return self._fetch_api(self.credentials)

        _api.refresh_api_callback = refresh_callback

        if self.credentials is None:
            raise ValueError(f"{self}'s credentials (signing key) is None!")

        APIRegistry.set_api_for(
            server_uid=self.id,
            user_verify_key=self.credentials.verify_key,
            api=_api,
        )

        self._api = _api
        self._api.metadata = self.metadata
        self.services = _api.services

        return _api


def connect(
    url: str | ServerURL = DEFAULT_SYFT_UI_ADDRESS,
    server: AbstractServer | None = None,
    port: int | None = None,
) -> SyftClient:
    if server:
        connection = PythonConnection(server=server)
    else:
        url = ServerURL.from_url(url)
        if isinstance(port, int | str):
            url.set_port(int(port))
        connection = HTTPConnection(url=url)

    client_type = connection.get_client_type().unwrap()
    return client_type(connection=connection)


def register(
    url: str | ServerURL,
    port: int,
    name: str,
    email: str,
    password: str,
    institution: str | None = None,
    website: str | None = None,
) -> SyftSigningKey | None:
    guest_client = connect(url=url, port=port)
    return guest_client.register(
        name=name,
        email=email,
        password=password,
        institution=institution,
        website=website,
    )


def login_as_guest(
    # HTTPConnection
    url: str | ServerURL = DEFAULT_SYFT_UI_ADDRESS,
    port: int | None = None,
    # PythonConnection
    server: AbstractServer | None = None,
    verbose: bool = True,
) -> SyftClient:
    _client = connect(
        url=url,
        server=server,
        port=port,
    )

    if verbose and _client.metadata is not None:
        print(
            f"Logged into <{_client.name}: {_client.metadata.server_side_type.capitalize()}-"
            f"side {_client.metadata.server_type.capitalize()}> as GUEST"
        )

    return _client.guest()


def login(
    email: str,
    # HTTPConnection
    url: str | ServerURL = DEFAULT_SYFT_UI_ADDRESS,
    port: int | None = None,
    # PythonConnection
    server: AbstractServer | None = None,
    password: str | None = None,
    cache: bool = True,
) -> SyftClient:
    _client = connect(
        url=url,
        server=server,
        port=port,
    )

    connection = _client.connection

    login_credentials = None

    if not password:
        password = getpass("Password: ")
    login_credentials = UserLoginCredentials(email=email, password=password)

    if cache and login_credentials:
        _client_cache = SyftClientSessionCache.get_client(
            login_credentials.email,
            login_credentials.password,
            connection=connection,
        )
        if _client_cache:
            print(
                f"Using cached client for {_client.name} as <{login_credentials.email}>"
            )
            _client = _client_cache

    if not _client.authed and login_credentials:
        _client = _client.login(
            email=login_credentials.email,
            password=login_credentials.password,
            cache=cache,
        )

    return _client


class SyftClientSessionCache:
    __credentials_store__: dict = {}
    __cache_key_format__ = "{email}-{password}-{connection}"
    __client_cache__: dict = {}

    @classmethod
    def _get_key(cls, email: str, password: str, connection: str) -> str:
        key = cls.__cache_key_format__.format(
            email=email, password=password, connection=connection
        )
        ph = PasswordHasher()
        return ph.hash(key)

    @classmethod
    def add_client(
        cls,
        email: str,
        password: str,
        connection: ServerConnection,
        syft_client: SyftClient,
    ) -> None:
        hash_key = cls._get_key(email, password, connection.get_cache_key())
        cls.__credentials_store__[hash_key] = syft_client
        cls.__client_cache__[syft_client.id] = syft_client

    @classmethod
    def add_client_by_uid_and_verify_key(
        cls,
        verify_key: SyftVerifyKey,
        server_uid: UID,
        syft_client: SyftClient,
    ) -> None:
        hash_key = str(server_uid) + str(verify_key)
        cls.__client_cache__[hash_key] = syft_client

    @classmethod
    def get_client_by_uid_and_verify_key(
        cls, verify_key: SyftVerifyKey, server_uid: UID
    ) -> SyftClient | None:
        hash_key = str(server_uid) + str(verify_key)
        return cls.__client_cache__.get(hash_key, None)

    @classmethod
    def get_client(
        cls, email: str, password: str, connection: ServerConnection
    ) -> SyftClient | None:
        # we have some bugs here so lets disable until they are fixed.
        return None
        # hash_key = cls._get_key(email, password, connection.get_cache_key())
        # return cls.__credentials_store__.get(hash_key, None)

    @classmethod
    def get_client_for_server_uid(cls, server_uid: UID) -> SyftClient | None:
        return cls.__client_cache__.get(server_uid, None)
