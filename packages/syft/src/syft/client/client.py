# future
from __future__ import annotations

# stdlib
from copy import deepcopy
from enum import Enum
from getpass import getpass
import json
import os
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from typing import Tuple
from typing import Type
from typing import Union
from typing import cast

# third party
from argon2 import PasswordHasher
import pydantic
import requests
from requests import Response
from requests import Session
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from typing_extensions import Self

# relative
from .. import __version__
from ..abstract_node import AbstractNode
from ..abstract_node import NodeSideType
from ..abstract_node import NodeType
from ..node.credentials import SyftSigningKey
from ..node.credentials import SyftVerifyKey
from ..node.credentials import UserLoginCredentials
from ..protocol.data_protocol import DataProtocol
from ..protocol.data_protocol import PROTOCOL_TYPE
from ..protocol.data_protocol import get_data_protocol
from ..serde.deserialize import _deserialize
from ..serde.serializable import serializable
from ..serde.serialize import _serialize
from ..service.context import NodeServiceContext
from ..service.metadata.node_metadata import NodeMetadataJSON
from ..service.metadata.node_metadata import NodeMetadataV3
from ..service.response import SyftError
from ..service.response import SyftSuccess
from ..service.user.user import UserCreate
from ..service.user.user import UserPrivateKey
from ..service.user.user import UserView
from ..service.user.user_roles import ServiceRole
from ..service.user.user_service import UserService
from ..types.grid_url import GridURL
from ..types.syft_object import SYFT_OBJECT_VERSION_1
from ..types.uid import UID
from ..util.logger import debug
from ..util.telemetry import instrument
from ..util.util import prompt_warning_message
from ..util.util import thread_ident
from ..util.util import verify_tls
from .api import APIModule
from .api import APIRegistry
from .api import SignedSyftAPICall
from .api import SyftAPI
from .api import SyftAPICall
from .api import debox_signed_syftapicall_response
from .connection import NodeConnection

if TYPE_CHECKING:
    # relative
    from ..service.network.node_peer import NodePeer

# use to enable mitm proxy
# from syft.grid.connections.http_connection import HTTPConnection
# HTTPConnection.proxies = {"http": "http://127.0.0.1:8080"}


def upgrade_tls(url: GridURL, response: Response) -> GridURL:
    try:
        if response.url.startswith("https://") and url.protocol == "http":
            # we got redirected to https
            https_url = GridURL.from_url(response.url).with_path("")
            debug(f"GridURL Upgraded to HTTPS. {https_url}")
            return https_url
    except Exception as e:
        print(f"Failed to upgrade to HTTPS. {e}")
    return url


def forward_message_to_proxy(
    make_call: Callable,
    proxy_target_uid: UID,
    path: str,
    credentials: Optional[SyftSigningKey] = None,
    args: Optional[Tuple] = None,
    kwargs: Optional[Dict] = None,
):
    kwargs = {} if kwargs is None else kwargs
    args = [] if args is None else args
    call = SyftAPICall(
        node_uid=proxy_target_uid,
        path=path,
        args=args,
        kwargs=kwargs,
        blocking=True,
    )

    if credentials is None:
        # generate a random signing key
        credentials = SyftSigningKey.generate()

    signed_message = call.sign(credentials=credentials)
    signed_result = make_call(signed_message)
    response = debox_signed_syftapicall_response(signed_result)
    return response


API_PATH = "/api/v2"
DEFAULT_PYGRID_PORT = 80
DEFAULT_PYGRID_ADDRESS = f"http://localhost:{DEFAULT_PYGRID_PORT}"


class Routes(Enum):
    ROUTE_METADATA = f"{API_PATH}/metadata"
    ROUTE_API = f"{API_PATH}/api"
    ROUTE_LOGIN = f"{API_PATH}/login"
    ROUTE_REGISTER = f"{API_PATH}/register"
    ROUTE_API_CALL = f"{API_PATH}/api_call"
    ROUTE_BLOB_STORE = "/blob"


@serializable(attrs=["proxy_target_uid", "url"])
class HTTPConnection(NodeConnection):
    __canonical_name__ = "HTTPConnection"
    __version__ = SYFT_OBJECT_VERSION_1

    proxy_target_uid: Optional[UID]
    url: GridURL
    routes: Type[Routes] = Routes
    session_cache: Optional[Session]

    @pydantic.validator("url", pre=True, always=True)
    def make_url(cls, v: Union[GridURL, str]) -> GridURL:
        return GridURL.from_url(v).as_container_host()

    def with_proxy(self, proxy_target_uid: UID) -> Self:
        return HTTPConnection(url=self.url, proxy_target_uid=proxy_target_uid)

    def get_cache_key(self) -> str:
        return str(self.url)

    @property
    def api_url(self) -> GridURL:
        return self.url.with_path(self.routes.ROUTE_API_CALL.value)

    def to_blob_route(self, path: str, **kwargs) -> GridURL:
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

    def _make_get(self, path: str, params: Optional[Dict] = None) -> bytes:
        url = self.url.with_path(path)
        response = self.session.get(
            str(url), verify=verify_tls(), proxies={}, params=params
        )
        if response.status_code != 200:
            raise requests.ConnectionError(
                f"Failed to fetch {url}. Response returned with code {response.status_code}"
            )

        # upgrade to tls if available
        self.url = upgrade_tls(self.url, response)

        return response.content

    def _make_post(
        self,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[bytes] = None,
    ) -> bytes:
        url = self.url.with_path(path)
        response = self.session.post(
            str(url), verify=verify_tls(), json=json, proxies={}, data=data
        )
        if response.status_code != 200:
            raise requests.ConnectionError(
                f"Failed to fetch {url}. Response returned with code {response.status_code}"
            )

        # upgrade to tls if available
        self.url = upgrade_tls(self.url, response)

        return response.content

    def get_node_metadata(self, credentials: SyftSigningKey) -> NodeMetadataJSON:
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
            return NodeMetadataJSON(**metadata_json)

    def get_api(
        self, credentials: SyftSigningKey, communication_protocol: int
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
        if self.proxy_target_uid:
            obj.node_uid = self.proxy_target_uid
        return cast(SyftAPI, obj)

    def login(
        self,
        email: str,
        password: str,
    ) -> Optional[SyftSigningKey]:
        credentials = {"email": email, "password": password}
        if self.proxy_target_uid:
            obj = forward_message_to_proxy(
                self.make_call,
                proxy_target_uid=self.proxy_target_uid,
                path="login",
                kwargs=credentials,
            )
        else:
            response = self._make_post(self.routes.ROUTE_LOGIN.value, credentials)
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
        return response

    def make_call(self, signed_call: SignedSyftAPICall) -> Union[Any, SyftError]:
        msg_bytes: bytes = _serialize(obj=signed_call, to_bytes=True)
        response = requests.post(  # nosec
            url=str(self.api_url),
            data=msg_bytes,
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

    def get_client_type(self) -> Type[SyftClient]:
        # TODO: Rasswanth, should remove passing in credentials
        # when metadata are proxy forwarded in the grid routes
        # in the gateway fixes PR
        # relative
        from .domain_client import DomainClient
        from .enclave_client import EnclaveClient
        from .gateway_client import GatewayClient

        metadata = self.get_node_metadata(credentials=SyftSigningKey.generate())
        if metadata.node_type == NodeType.DOMAIN.value:
            return DomainClient
        elif metadata.node_type == NodeType.GATEWAY.value:
            return GatewayClient
        elif metadata.node_type == NodeType.ENCLAVE.value:
            return EnclaveClient
        else:
            return SyftError(message=f"Unknown node type {metadata.node_type}")


@serializable()
class PythonConnection(NodeConnection):
    __canonical_name__ = "PythonConnection"
    __version__ = SYFT_OBJECT_VERSION_1

    node: AbstractNode
    proxy_target_uid: Optional[UID]

    def with_proxy(self, proxy_target_uid: UID) -> Self:
        return PythonConnection(node=self.node, proxy_target_uid=proxy_target_uid)

    def get_node_metadata(self, credentials: SyftSigningKey) -> NodeMetadataJSON:
        if self.proxy_target_uid:
            response = forward_message_to_proxy(
                make_call=self.make_call,
                proxy_target_uid=self.proxy_target_uid,
                path="metadata",
                credentials=credentials,
            )
            return response
        else:
            return self.node.metadata.to(NodeMetadataJSON)

    def to_blob_route(self, path: str, host=None) -> GridURL:
        # TODO: FIX!
        if host is not None:
            return GridURL(host_or_ip=host, port=8333).with_path(path)
        else:
            return GridURL(port=8333).with_path(path)

    def get_api(
        self, credentials: SyftSigningKey, communication_protocol: int
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
            obj = self.node.get_api(
                for_user=credentials.verify_key,
                communication_protocol=communication_protocol,
            )
        obj.connection = self
        obj.signing_key = credentials
        obj.communication_protocol = communication_protocol
        if self.proxy_target_uid:
            obj.node_uid = self.proxy_target_uid
        return obj

    def get_cache_key(self) -> str:
        return str(self.node.id)

    def exchange_credentials(
        self, email: str, password: str
    ) -> Optional[UserPrivateKey]:
        context = self.node.get_unauthed_context(
            login_credentials=UserLoginCredentials(email=email, password=password)
        )
        method = self.node.get_method_with_context(
            UserService.exchange_credentials, context
        )
        result = method()
        return result

    def login(
        self,
        email: str,
        password: str,
    ) -> Optional[SyftSigningKey]:
        if self.proxy_target_uid:
            obj = forward_message_to_proxy(
                self.make_call,
                proxy_target_uid=self.proxy_target_uid,
                path="login",
                kwargs={"email": email, "password": password},
            )

        else:
            obj = self.exchange_credentials(email=email, password=password)
        return obj

    def register(self, new_user: UserCreate) -> Optional[SyftSigningKey]:
        if self.proxy_target_uid:
            response = forward_message_to_proxy(
                self.make_call,
                proxy_target_uid=self.proxy_target_uid,
                path="register",
                kwargs={"new_user": new_user},
            )
        else:
            service_context = NodeServiceContext(node=self.node)
            method = self.node.get_service_method(UserService.register)
            response = method(context=service_context, new_user=new_user)
        return response

    def make_call(self, signed_call: SignedSyftAPICall) -> Union[Any, SyftError]:
        return self.node.handle_api_call(signed_call)

    def __repr__(self) -> str:
        return f"{type(self).__name__}"

    def __str__(self) -> str:
        return f"{type(self).__name__}"

    def get_client_type(self) -> Type[SyftClient]:
        # relative
        from .domain_client import DomainClient
        from .enclave_client import EnclaveClient
        from .gateway_client import GatewayClient

        metadata = self.get_node_metadata(credentials=SyftSigningKey.generate())
        if metadata.node_type == NodeType.DOMAIN.value:
            return DomainClient
        elif metadata.node_type == NodeType.GATEWAY.value:
            return GatewayClient
        elif metadata.node_type == NodeType.ENCLAVE.value:
            return EnclaveClient
        else:
            return SyftError(message=f"Unknown node type {metadata.node_type}")


@instrument
@serializable()
class SyftClient:
    connection: NodeConnection
    metadata: Optional[NodeMetadataJSON]
    credentials: Optional[SyftSigningKey]
    __logged_in_user: str = ""
    __logged_in_username: str = ""
    __user_role: ServiceRole = ServiceRole.NONE

    def __init__(
        self,
        connection: NodeConnection,
        metadata: Optional[NodeMetadataJSON] = None,
        credentials: Optional[SyftSigningKey] = None,
        api: Optional[SyftAPI] = None,
    ) -> None:
        self.connection = connection
        self.metadata = metadata
        self.credentials: Optional[SyftSigningKey] = credentials
        self._api = api
        self.communication_protocol = None
        self.current_protocol = None

        self.post_init()

    def get_env(self) -> str:
        return self.api.services.metadata.get_env()

    def post_init(self) -> None:
        if self.metadata is None:
            self._fetch_node_metadata(self.credentials)

        self.communication_protocol = self._get_communication_protocol(
            self.metadata.supported_protocols
        )

    def _get_communication_protocol(
        self, protocols_supported_by_server: List
    ) -> Union[int, str]:
        data_protocol: DataProtocol = get_data_protocol()
        protocols_supported_by_client: List[
            PROTOCOL_TYPE
        ] = data_protocol.supported_protocols

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
        project = project_create.start()
        return project

    def sync_code_from_request(self, request):
        # relative
        from ..service.code.user_code import UserCode
        from ..store.linked_obj import LinkedObject

        code: Union[UserCode, SyftError] = request.code
        if isinstance(code, SyftError):
            return code

        code = deepcopy(code)
        code.node_uid = self.id
        code.user_verify_key = self.verify_key

        def get_nested_codes(code: UserCode):
            result = []
            for __, (linked_code_obj, _) in code.nested_codes.items():
                nested_code = linked_code_obj.resolve
                nested_code = deepcopy(nested_code)
                nested_code.node_uid = code.node_uid
                nested_code.user_verify_key = code.user_verify_key
                result.append(nested_code)
                result += get_nested_codes(nested_code)

            updated_code_links = {
                nested_code.service_func_name: (LinkedObject.from_obj(nested_code), {})
                for nested_code in result
            }
            code.nested_codes = updated_code_links
            return result

        nested_codes = get_nested_codes(code)

        for c in nested_codes + [code]:
            res = self.code.submit(c)
            if isinstance(res, SyftError):
                return res
        self._fetch_api(self.credentials)
        return SyftSuccess(message="User Code Submitted")

    @property
    def authed(self) -> bool:
        return bool(self.credentials)

    @property
    def logged_in_user(self) -> Optional[str]:
        return self.__logged_in_user

    @property
    def logged_in_username(self) -> Optional[str]:
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
    def from_url(cls, url: Union[str, GridURL]) -> Self:
        return cls(connection=HTTPConnection(GridURL.from_url(url)))

    @classmethod
    def from_node(cls, node: AbstractNode) -> Self:
        return cls(connection=PythonConnection(node=node))

    @property
    def name(self) -> Optional[str]:
        return self.metadata.name if self.metadata else None

    @property
    def id(self) -> Optional[UID]:
        return UID.from_string(self.metadata.id) if self.metadata else None

    @property
    def icon(self) -> str:
        return "📡"

    @property
    def peer(self) -> Any:
        # relative
        from ..service.network.network_service import NodePeer

        return NodePeer.from_client(self)

    @property
    def route(self) -> Any:
        return self.connection.route

    @property
    def api(self) -> SyftAPI:
        # invalidate API
        if self._api is None or (self._api.signing_key != self.credentials):
            self._fetch_api(self.credentials)

        return self._api

    def guest(self) -> Self:
        return self.__class__(
            connection=self.connection,
            credentials=SyftSigningKey.generate(),
            metadata=self.metadata,
        )

    def exchange_route(self, client: Self) -> Union[SyftSuccess, SyftError]:
        # relative
        from ..service.network.routes import connection_to_route

        self_node_route = connection_to_route(self.connection)
        remote_node_route = connection_to_route(client.connection)

        result = self.api.services.network.exchange_credentials_with(
            self_node_route=self_node_route,
            remote_node_route=remote_node_route,
            remote_node_verify_key=client.metadata.to(NodeMetadataV3).verify_key,
        )

        return result

    @property
    def jobs(self) -> Optional[APIModule]:
        if self.api.has_service("job"):
            return self.api.services.job
        return None

    @property
    def users(self) -> Optional[APIModule]:
        if self.api.has_service("user"):
            return self.api.services.user
        return None

    @property
    def numpy(self) -> Optional[APIModule]:
        if self.api.has_lib("numpy"):
            return self.api.lib.numpy
        return None

    @property
    def settings(self) -> Optional[APIModule]:
        if self.api.has_service("user"):
            return self.api.services.settings
        return None

    @property
    def notifications(self) -> Optional[APIModule]:
        print(
            "WARNING: Notifications is currently is in a beta state, so use carefully!"
        )
        print("If possible try using client.requests/client.projects")
        if self.api.has_service("notifications"):
            return self.api.services.notifications
        return None

    @property
    def peers(self) -> Optional[Union[List[NodePeer], SyftError]]:
        if self.api.has_service("network"):
            return self.api.services.network.get_all_peers()
        return None

    @property
    def me(self) -> Optional[Union[UserView, SyftError]]:
        if self.api.has_service("user"):
            return self.api.services.user.get_current_user()
        return None

    def login_as_guest(self) -> Self:
        _guest_client = self.guest()

        print(
            f"Logged into <{self.name}: {self.metadata.node_side_type.capitalize()}-side "
            f"{self.metadata.node_type.capitalize()}> as GUEST"
        )

        return _guest_client

    def login(
        self,
        email: Optional[str] = None,
        password: Optional[str] = None,
        cache: bool = True,
        register: bool = False,
        **kwargs: Any,
    ) -> Self:
        # If SYFT_LOGIN_{NODE_NAME}_PASSWORD is set, use that as the password
        # for the login. This is useful for CI/CD environments to test password
        # randomization that is implemented by helm charts
        if self.name is not None and email == "info@openmined.org":
            pass_env_var = f"SYFT_LOGIN_{self.name}_PASSWORD"
            if pass_env_var in os.environ:
                password = os.environ[pass_env_var]

        if email is None:
            email = input("Email: ")
        if password is None:
            password = getpass("Password: ")

        if register:
            self.register(
                email=email, password=password, password_verify=password, **kwargs
            )

        user_private_key = self.connection.login(email=email, password=password)
        if isinstance(user_private_key, SyftError):
            return user_private_key

        signing_key = None if user_private_key is None else user_private_key.signing_key

        client = self.__class__(
            connection=self.connection,
            metadata=self.metadata,
            credentials=signing_key,
        )

        client.__logged_in_user = email

        if user_private_key is not None:
            client.__user_role = user_private_key.role
            client.__logged_in_username = client.users.get_current_user().name

        if signing_key is not None:
            print(
                f"Logged into <{client.name}: {client.metadata.node_side_type.capitalize()} side "
                f"{client.metadata.node_type.capitalize()}> as <{email}>"
            )
            # relative
            from ..node.node import get_default_root_password

            if password == get_default_root_password():
                message = (
                    "You are using a default password. Please change the password "
                    "using `[your_client].me.set_password([new_password])`."
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
                # node uid and verify key are not individually unique
                # both the combination of node uid and verify key are unique
                # which could be used to identity a client uniquely of any given node
                # TODO: It would be better to have a single cache storage
                # combining both email, password and verify key and uid
                SyftClientSessionCache.add_client_by_uid_and_verify_key(
                    verify_key=signing_key.verify_key,
                    node_uid=client.id,
                    syft_client=client,
                )

        # relative
        from ..node.node import CODE_RELOADER

        CODE_RELOADER[thread_ident()] = client._reload_user_code

        return client

    def _reload_user_code(self):
        # relative
        from ..service.code.user_code import load_approved_policy_code

        user_code_items = self.code.get_all_for_user()
        load_approved_policy_code(user_code_items)

    def register(
        self,
        name: str,
        email: Optional[str] = None,
        password: Optional[str] = None,
        password_verify: Optional[str] = None,
        institution: Optional[str] = None,
        website: Optional[str] = None,
    ):
        if not email:
            email = input("Email: ")
        if not password:
            password = getpass("Password: ")
        if not password_verify:
            password_verify = getpass("Confirm Password: ")
        if password != password_verify:
            return SyftError(message="Passwords do not match")

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
            return SyftError(message=str(e))

        if self.metadata.node_side_type == NodeSideType.HIGH_SIDE.value:
            message = (
                "You're registering a user to a high side "
                f"{self.metadata.node_type}, which could "
                "host datasets with private information."
            )
            if self.metadata.show_warnings and not prompt_warning_message(
                message=message
            ):
                return

        response = self.connection.register(new_user=new_user)
        if isinstance(response, tuple):
            response = response[0]
        return response

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

    def _fetch_node_metadata(self, credentials: SyftSigningKey) -> None:
        metadata = self.connection.get_node_metadata(credentials=credentials)
        if isinstance(metadata, NodeMetadataJSON):
            metadata.check_version(__version__)
            self.metadata = metadata

    def _fetch_api(self, credentials: SyftSigningKey):
        _api: SyftAPI = self.connection.get_api(
            credentials=credentials,
            communication_protocol=self.communication_protocol,
        )

        def refresh_callback():
            return self._fetch_api(self.credentials)

        _api.refresh_api_callback = refresh_callback
        APIRegistry.set_api_for(
            node_uid=self.id,
            user_verify_key=self.credentials.verify_key,
            api=_api,
        )
        self._api = _api


@instrument
def connect(
    url: Union[str, GridURL] = DEFAULT_PYGRID_ADDRESS,
    node: Optional[AbstractNode] = None,
    port: Optional[int] = None,
) -> SyftClient:
    if node:
        connection = PythonConnection(node=node)
    else:
        url = GridURL.from_url(url)
        if isinstance(port, (int, str)):
            url.set_port(int(port))
        connection = HTTPConnection(url=url)

    client_type = connection.get_client_type()

    if isinstance(client_type, SyftError):
        return client_type

    return client_type(connection=connection)


@instrument
def register(
    url: Union[str, GridURL],
    port: int,
    name: str,
    email: str,
    password: str,
    institution: Optional[str] = None,
    website: Optional[str] = None,
):
    guest_client = connect(url=url, port=port)
    return guest_client.register(
        name=name,
        email=email,
        password=password,
        institution=institution,
        website=website,
    )


@instrument
def login_as_guest(
    url: Union[str, GridURL] = DEFAULT_PYGRID_ADDRESS,
    node: Optional[AbstractNode] = None,
    port: Optional[int] = None,
    verbose: bool = True,
):
    _client = connect(url=url, node=node, port=port)

    if isinstance(_client, SyftError):
        return _client

    if verbose:
        print(
            f"Logged into <{_client.name}: {_client.metadata.node_side_type.capitalize()}-"
            f"side {_client.metadata.node_type.capitalize()}> as GUEST"
        )

    return _client.guest()


@instrument
def login(
    email: str,
    url: Union[str, GridURL] = DEFAULT_PYGRID_ADDRESS,
    node: Optional[AbstractNode] = None,
    port: Optional[int] = None,
    password: Optional[str] = None,
    cache: bool = True,
) -> SyftClient:
    _client = connect(url=url, node=node, port=port)

    if isinstance(_client, SyftError):
        return _client

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
    __credentials_store__: Dict = {}
    __cache_key_format__ = "{email}-{password}-{connection}"
    __client_cache__: Dict = {}

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
        connection: NodeConnection,
        syft_client: SyftClient,
    ):
        hash_key = cls._get_key(email, password, connection.get_cache_key())
        cls.__credentials_store__[hash_key] = syft_client
        cls.__client_cache__[syft_client.id] = syft_client

    @classmethod
    def add_client_by_uid_and_verify_key(
        cls,
        verify_key: SyftVerifyKey,
        node_uid: UID,
        syft_client: SyftClient,
    ):
        hash_key = str(node_uid) + str(verify_key)
        cls.__client_cache__[hash_key] = syft_client

    @classmethod
    def get_client_by_uid_and_verify_key(
        cls, verify_key: SyftVerifyKey, node_uid: UID
    ) -> Optional[SyftClient]:
        hash_key = str(node_uid) + str(verify_key)
        return cls.__client_cache__.get(hash_key, None)

    @classmethod
    def get_client(
        cls, email: str, password: str, connection: NodeConnection
    ) -> Optional[SyftClient]:
        # we have some bugs here so lets disable until they are fixed.
        return None
        hash_key = cls._get_key(email, password, connection.get_cache_key())
        return cls.__credentials_store__.get(hash_key, None)

    @classmethod
    def get_client_for_node_uid(cls, node_uid: UID) -> Optional[SyftClient]:
        return cls.__client_cache__.get(node_uid, None)
