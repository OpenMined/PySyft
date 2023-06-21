# future
from __future__ import annotations

# stdlib
from enum import Enum
import hashlib
import json
from typing import Any
from typing import Dict
from typing import Optional
from typing import TYPE_CHECKING
from typing import Type
from typing import Union
from typing import cast

# third party
import pydantic
import requests
from requests import Response
from requests import Session
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from tqdm import tqdm
from typing_extensions import Self

# relative
from .. import __version__
from ..abstract_node import AbstractNode
from ..img.base64 import base64read
from ..node.credentials import SyftSigningKey
from ..node.credentials import SyftVerifyKey
from ..node.credentials import UserLoginCredentials
from ..serde.deserialize import _deserialize
from ..serde.serializable import serializable
from ..serde.serialize import _serialize
from ..service.context import NodeServiceContext
from ..service.dataset.dataset import CreateDataset
from ..service.metadata.node_metadata import NodeMetadata
from ..service.metadata.node_metadata import NodeMetadataJSON
from ..service.response import SyftError
from ..service.response import SyftSuccess
from ..service.user.user import UserCreate
from ..service.user.user import UserPrivateKey
from ..service.user.user_roles import ServiceRole
from ..service.user.user_service import UserService
from ..types.grid_url import GridURL
from ..types.syft_object import SYFT_OBJECT_VERSION_1
from ..types.uid import UID
from ..util.fonts import fonts_css
from ..util.logger import debug
from ..util.telemetry import instrument
from ..util.util import thread_ident
from ..util.util import verify_tls
from .api import APIModule
from .api import APIRegistry
from .api import SignedSyftAPICall
from .api import SyftAPI
from .api import SyftAPICall
from .connection import NodeConnection

# use to enable mitm proxy
# from syft.grid.connections.http_connection import HTTPConnection
# HTTPConnection.proxies = {"http": "http://127.0.0.1:8080"}

if TYPE_CHECKING:
    # relative
    from ..service.project.project import Project


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


API_PATH = "/api/v2"
DEFAULT_PYGRID_PORT = 80
DEFAULT_PYGRID_ADDRESS = f"http://localhost:{DEFAULT_PYGRID_PORT}"


class Routes(Enum):
    ROUTE_METADATA = f"{API_PATH}/metadata"
    ROUTE_API = f"{API_PATH}/api"
    ROUTE_LOGIN = f"{API_PATH}/login"
    ROUTE_REGISTER = f"{API_PATH}/register"
    ROUTE_API_CALL = f"{API_PATH}/api_call"


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
            call = SyftAPICall(
                node_uid=self.proxy_target_uid,
                path="metadata",
                args=[],
                kwargs={},
                blocking=True,
            )
            signed_call = call.sign(credentials=credentials)
            response = self.make_call(signed_call)
            if isinstance(response, SyftError):
                return response
            return response.to(NodeMetadataJSON)
        else:
            response = self._make_get(self.routes.ROUTE_METADATA.value)
            metadata_json = json.loads(response)
            return NodeMetadataJSON(**metadata_json)

    def get_api(self, credentials: SyftSigningKey) -> SyftAPI:
        params = {"verify_key": str(credentials.verify_key)}
        content = self._make_get(self.routes.ROUTE_API.value, params=params)
        obj = _deserialize(content, from_bytes=True)
        obj.connection = self
        obj.signing_key = credentials
        if self.proxy_target_uid:
            obj.node_uid = self.proxy_target_uid
        return cast(SyftAPI, obj)

    def login(self, email: str, password: str) -> Optional[SyftSigningKey]:
        credentials = {"email": email, "password": password}
        response = self._make_post(self.routes.ROUTE_LOGIN.value, credentials)
        obj = _deserialize(response, from_bytes=True)
        if isinstance(obj, UserPrivateKey):
            return obj
        return None

    def register(self, new_user: UserCreate) -> SyftSigningKey:
        data = _serialize(new_user, to_bytes=True)
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
            call = SyftAPICall(
                node_uid=self.proxy_target_uid,
                path="metadata",
                args=[],
                kwargs={},
                blocking=True,
            )
            signed_call = call.sign(credentials=credentials)
            response = self.make_call(signed_call)
            if isinstance(response, SyftError):
                return response
            return response.to(NodeMetadataJSON)
        else:
            return self.node.metadata.to(NodeMetadataJSON)

    def get_api(self, credentials: SyftSigningKey) -> SyftAPI:
        # todo: its a bit odd to identify a user by its verify key maybe?
        obj = self.node.get_api(for_user=credentials.verify_key)
        obj.connection = self
        obj.signing_key = credentials
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

    def login(self, email: str, password: str) -> Optional[SyftSigningKey]:
        obj = self.exchange_credentials(email=email, password=password)
        if isinstance(obj, UserPrivateKey):
            return obj
        return None

    def register(self, new_user: UserCreate) -> Optional[SyftSigningKey]:
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


@instrument
@serializable()
class SyftClient:
    connection: NodeConnection
    metadata: Optional[NodeMetadataJSON]
    credentials: Optional[SyftSigningKey]
    __logged_in_user: str = ""
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

        self.post_init()

    def post_init(self) -> None:
        if self.metadata is None:
            self._fetch_node_metadata(self.credentials)

    @property
    def authed(self) -> bool:
        return bool(self.credentials)

    @property
    def logged_in_user(self) -> Optional[str]:
        return self.__logged_in_user

    @property
    def user_role(self) -> ServiceRole:
        return self.__user_role

    @property
    def verify_key(self) -> SyftVerifyKey:
        if self.credentials is None:
            raise ValueError("SigningKey not set on client")
        return self.credentials.verify_key

    @staticmethod
    def from_url(url: Union[str, GridURL]) -> Self:
        return SyftClient(connection=HTTPConnection(GridURL.from_url(url)))

    @staticmethod
    def from_node(node: AbstractNode) -> Self:
        return SyftClient(connection=PythonConnection(node=node))

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
    def api(self) -> SyftAPI:
        # invalidate API
        if self._api is None or (self._api.signing_key != self.credentials):
            self._fetch_api(self.credentials)

        return self._api

    def guest(self) -> Self:
        return SyftClient(
            connection=self.connection,
            credentials=SyftSigningKey.generate(),
            metadata=self.metadata,
        )

    def upload_dataset(self, dataset: CreateDataset) -> Union[SyftSuccess, SyftError]:
        # relative
        from ..types.twin_object import TwinObject

        dataset._check_asset_must_contain_mock()

        for asset in tqdm(dataset.asset_list):
            print(f"Uploading: {asset.name}")
            try:
                twin = TwinObject(private_obj=asset.data, mock_obj=asset.mock)
            except Exception as e:
                return SyftError(message=f"Failed to create twin. {e}")
            response = self.api.services.action.set(twin)
            if isinstance(response, SyftError):
                print(f"Failed to upload asset\n: {asset}")
                return response
            asset.action_id = twin.id
            asset.node_uid = self.id
        valid = dataset.check()
        if valid.ok():
            return self.api.services.dataset.add(dataset=dataset)
        else:
            if len(valid.err()) > 0:
                return tuple(valid.err())
            return valid.err()

    def exchange_route(self, client: Self) -> Union[SyftSuccess, SyftError]:
        # relative
        from ..service.network.routes import connection_to_route

        self_node_route = connection_to_route(self.connection)
        remote_node_route = connection_to_route(client.connection)

        result = self.api.services.network.exchange_credentials_with(
            self_node_route=self_node_route,
            remote_node_route=remote_node_route,
            remote_node_verify_key=client.metadata.to(NodeMetadata).verify_key,
        )

        return result

    def apply_to_gateway(self, client: Self) -> None:
        return self.exchange_route(client)

    @property
    def data_subject_registry(self) -> Optional[APIModule]:
        if self.api is not None and self.api.has_service("data_subject"):
            return self.api.services.data_subject
        return None

    @property
    def users(self) -> Optional[APIModule]:
        if self.api is not None and self.api.has_service("user"):
            return self.api.services.user
        return None

    @property
    def settings(self) -> Optional[APIModule]:
        if self.api is not None and self.api.has_service("user"):
            return self.api.services.settings
        return None

    @property
    def code(self) -> Optional[APIModule]:
        if self.api is not None and self.api.has_service("code"):
            return self.api.services.code

    @property
    def requests(self) -> Optional[APIModule]:
        if self.api is not None and self.api.has_service("request"):
            return self.api.services.request
        return None

    @property
    def datasets(self) -> Optional[APIModule]:
        if self.api is not None and self.api.has_service("dataset"):
            return self.api.services.dataset
        return None

    @property
    def notifications(self) -> Optional[APIModule]:
        print(
            "WARNING: Notifications is currently is in a beta state, so use carefully!"
        )
        print("If possible try using client.requests/client.projects")
        if self.api is not None and self.api.has_service("messages"):
            return self.api.services.messages
        return None

    @property
    def domains(self) -> Optional[APIModule]:
        if self.api is not None and self.api.has_service("network"):
            return self.api.services.network.get_all_peers()
        return None

    @property
    def projects(self) -> Optional[APIModule]:
        if self.api.has_service("project"):
            return self.api.services.project
        return None

    def get_project(
        self,
        name: str = None,
        uid: UID = None,
    ) -> Optional[Project]:
        """Get project by name or UID"""

        if not self.api.has_service("project"):
            return None

        if name:
            return self.api.services.project.get_by_name(name)

        elif uid:
            return self.api.services.project.get_by_uid(uid)

        return self.api.services.project.get_all()

    def login(self, email: str, password: str, cache: bool = True) -> Self:
        user_private_key = self.connection.login(email=email, password=password)
        signing_key = None
        if user_private_key is not None:
            signing_key = user_private_key.signing_key
            self.__user_role = user_private_key.role
        if signing_key is not None:
            self.credentials = signing_key
            self.__logged_in_user = email
            # TODO: How to get the role of the user?
            # self.__user_role =
            self._fetch_api(self.credentials)
            print(f"Logged into {self.name} as <{email}>")
            if cache:
                SyftClientSessionCache.add_client(
                    email=email,
                    password=password,
                    connection=self.connection,
                    syft_client=self,
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
                    node_uid=self.id,
                    syft_client=self,
                )

        # relative
        from ..node.node import CODE_RELOADER

        CODE_RELOADER[thread_ident()] = self._reload_user_code

        return self

    def _reload_user_code(self):
        # relative
        from ..service.code.user_code import load_approved_policy_code

        user_code_items = self.code.get_all_for_user()
        load_approved_policy_code(user_code_items)

    def register(
        self,
        name: str,
        email: str,
        password: str,
        institution: Optional[str] = None,
        website: Optional[str] = None,
    ):
        try:
            new_user = UserCreate(
                name=name,
                email=email,
                password=password,
                password_verify=password,
                institution=institution,
                website=website,
                created_by=self.credentials,
            )
        except Exception as e:
            return SyftError(message=str(e))
        response = self.connection.register(new_user=new_user)
        if isinstance(response, tuple):
            response = response[0]
        return response

    @property
    def peer(self) -> Any:
        # relative
        from ..service.network.network_service import NodePeer

        return NodePeer.from_client(self)

    @property
    def route(self) -> Any:
        return self.connection.route

    def proxy_to(self, peer: Any) -> Self:
        connection = self.connection.with_proxy(peer.id)
        client = SyftClient(
            connection=connection,
            credentials=self.credentials,
        )
        return client

    def __getattr__(self, name):
        if (
            hasattr(self, "api")
            and hasattr(self.api, "lib")
            and hasattr(self.api.lib, name)
        ):
            return getattr(self.api.lib, name)
        else:
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute {name}."
            )

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
            return f"<{client_type} - <{uid}>: via {self.id} {self.connection}>"
        return f"<{client_type} - {self.name} <{uid}>: {self.connection}>"

    def _repr_html_(self) -> str:
        guest_commands = """
        <li><span class='syft-code-block'>&lt;your_client&gt;.datasets</span> - list datasets</li>
        <li><span class='syft-code-block'>&lt;your_client&gt;.code</span> - list code</li>
        <li><span class='syft-code-block'>&lt;your_client&gt;.login</span> - list projects</li>
        <li>
            <span class='syft-code-block'>&lt;your_client&gt;.code.submit?</span> - display function signature
        </li>"""
        ds_commands = """
        <li><span class='syft-code-block'>&lt;your_client&gt;.datasets</span> - list datasets</li>
        <li><span class='syft-code-block'>&lt;your_client&gt;.code</span> - list code</li>
        <li><span class='syft-code-block'>&lt;your_client&gt;.projects</span> - list projects</li>
        <li>
            <span class='syft-code-block'>&lt;your_client&gt;.code.submit?</span> - display function signature
        </li>"""

        do_commands = """
        <li><span class='syft-code-block'>&lt;your_client&gt;.projects</span> - list projects</li>
        <li><span class='syft-code-block'>&lt;your_client&gt;.requests</span> - list requests</li>
        <li><span class='syft-code-block'>&lt;your_client&gt;.users</span> - list users</li>
        <li>
            <span class='syft-code-block'>&lt;your_client&gt;.requests.submit?</span> - display function signature
        </li>"""

        # TODO: how to select ds/do commands based on self.__user_role

        if (
            self.user_role.value == ServiceRole.NONE.value
            or self.user_role.value == ServiceRole.GUEST.value
        ):
            commands = guest_commands
        elif (
            self.user_role is not None
            and self.user_role.value == ServiceRole.DATA_SCIENTIST.value
        ):
            commands = ds_commands
        elif (
            self.user_role is not None
            and self.user_role.value >= ServiceRole.DATA_OWNER.value
        ):
            commands = do_commands

        command_list = f"""
        <ul style='padding-left: 1em;'>
            {commands}
        </ul>
        """

        small_grid_symbol_logo = base64read("small-grid-symbol-logo.png")

        return f"""
        <style>
            {fonts_css}

            .syft-container {{
                padding: 5px;
                font-family: 'Open Sans';
            }}
            .syft-alert-info {{
                color: #1F567A;
                background-color: #C2DEF0;
                border-radius: 4px;
                padding: 5px;
                padding: 13px 10px
            }}
            .syft-code-block {{
                background-color: #f7f7f7;
                border: 1px solid #cfcfcf;
                padding: 0px 2px;
            }}
            .syft-space {{
                margin-top: 1em;
            }}
        </style>
        <div class="syft-client syft-container">
            <img src="{small_grid_symbol_logo}" alt="Logo"
            style="width:48px;height:48px;padding:3px;">
            <h2>Welcome to {self.name}</h2>
            <div class="syft-space">
                <!-- <strong>Institution:</strong> TODO<br /> -->
                <!-- <strong>Owner:</strong> TODO<br /> -->
                <strong>URL:</strong> {getattr(self.connection, 'url', '')}<br />
                <!-- <strong>PyGrid Admin:</strong> TODO<br /> -->
            </div>
            <div class='syft-alert-info syft-space'>
                &#9432;&nbsp;
                This domain is run by the library PySyft to learn more about how it works visit
                <a href="https://github.com/OpenMined/PySyft">github.com/OpenMined/PySyft</a>.
            </div>
            <h4>Commands to Get Started</h4>
            {command_list}
        </div><br />
        """

    def _fetch_node_metadata(self, credentials: SyftSigningKey) -> None:
        metadata = self.connection.get_node_metadata(credentials=credentials)
        if isinstance(metadata, NodeMetadataJSON):
            metadata.check_version(__version__)
            self.metadata = metadata

    def _fetch_api(self, credentials: SyftSigningKey):
        _api: SyftAPI = self.connection.get_api(credentials=credentials)

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
    _client = SyftClient(connection=connection)
    return _client


@instrument
def login(
    url: Union[str, GridURL] = DEFAULT_PYGRID_ADDRESS,
    node: Optional[AbstractNode] = None,
    port: Optional[int] = None,
    email: Optional[str] = None,
    password: Optional[str] = None,
    cache: bool = True,
) -> SyftClient:
    _client = connect(url=url, node=node, port=port)
    connection = _client.connection

    login_credentials = None
    if email and password:
        login_credentials = UserLoginCredentials(email=email, password=password)

    if login_credentials is None:
        print(f"Logged into {_client.name} as GUEST")
        return _client.guest()

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
        _client.login(
            email=login_credentials.email,
            password=login_credentials.password,
            cache=cache,
        )
        if not _client.authed:
            return SyftError(message=f"Failed to login as {login_credentials.email}")

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
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

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
