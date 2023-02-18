# stdlib
from enum import Enum
import hashlib
import json
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union
from typing import cast

# third party
import requests
from requests import Response
from requests import Session
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from result import OkErr
from tqdm import tqdm
from typing_extensions import Self

# relative
from .... import __version__
from ....grid import GridURL
from ....logger import debug
from ....telemetry import instrument
from ....util import verify_tls
from ...common.serde.deserialize import _deserialize
from ...common.serde.serializable import serializable
from ...common.serde.serialize import _serialize
from ...common.uid import UID
from ...node.new.credentials import UserLoginCredentials
from ...node.new.node_metadata import NodeMetadataJSON
from ...node.new.user import UserPrivateKey
from .api import APIModule
from .api import APIRegistry
from .api import SyftAPI
from .api import SyftAPICall
from .connection import NodeConnection
from .credentials import SyftSigningKey
from .dataset import CreateDataset
from .node import NewNode
from .response import SyftError
from .response import SyftSuccess
from .user_service import UserService

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


API_PATH = "/api/v1/new"


class Routes(Enum):
    ROUTE_METADATA = f"{API_PATH}/metadata"
    ROUTE_API = f"{API_PATH}/api"
    ROUTE_LOGIN = f"{API_PATH}/login"
    ROUTE_API_CALL = f"{API_PATH}/api_call"


DEFAULT_PYGRID_PORT = 80
DEFAULT_PYGRID_ADDRESS = f"http://localhost:{DEFAULT_PYGRID_PORT}"


@serializable(recursive_serde=True)
class HTTPConnection(NodeConnection):
    proxies: Dict[str, str] = {}
    url: GridURL
    routes: Routes = Routes
    _session: Optional[Session]

    def __init__(self, url: Union[GridURL, str]) -> None:
        self.url = GridURL.from_url(url)
        self._session = None

    def get_cache_key(self) -> str:
        return str(self.url)

    @property
    def api_url(self) -> GridURL:
        return self.url.with_path(self.routes.ROUTE_API_CALL.value)

    @property
    def session(self) -> Session:
        if self._session is None:
            session = requests.Session()
            retry = Retry(total=3, backoff_factor=0.5)
            adapter = HTTPAdapter(max_retries=retry)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            self._session = session
        return self._session

    def _make_get(self, path: str) -> bytes:
        url = self.url.with_path(path)
        response = self.session.get(
            str(url), verify=verify_tls(), proxies=HTTPConnection.proxies
        )
        if response.status_code != 200:
            raise requests.ConnectionError(
                f"Failed to fetch {url}. Response returned with code {response.status_code}"
            )

        # upgrade to tls if available
        self.url = upgrade_tls(self.url, response)

        return response.content

    def _make_post(self, path: str, json: Dict[str, Any]) -> bytes:
        url = self.url.with_path(path)
        response = self.session.post(
            str(url), verify=verify_tls(), json=json, proxies=HTTPConnection.proxies
        )
        if response.status_code != 200:
            raise requests.ConnectionError(
                f"Failed to fetch {url}. Response returned with code {response.status_code}"
            )

        # upgrade to tls if available
        self.url = upgrade_tls(self.url, response)

        return response.content

    def get_node_metadata(self) -> NodeMetadataJSON:
        response = self._make_get(self.routes.ROUTE_METADATA.value)
        metadata_json = json.loads(response)
        return NodeMetadataJSON(**metadata_json)

    def get_api(self, credentials: SyftSigningKey) -> SyftAPI:
        content = self._make_get(self.routes.ROUTE_API.value)
        obj = _deserialize(content, from_bytes=True)
        obj.connection = self
        obj.signing_key = credentials
        return cast(SyftAPI, obj)

    def connect(self, email: str, password: str) -> SyftSigningKey:
        credentials = {"email": email, "password": password}
        response = self._make_post(self.routes.ROUTE_LOGIN.value, credentials)
        obj = _deserialize(response, from_bytes=True)
        if isinstance(obj, UserPrivateKey):
            return obj.signing_key
        print(obj)
        return None

    def make_call(self, signed_call: SyftAPICall) -> Union[Any, SyftError]:
        msg_bytes: bytes = _serialize(obj=signed_call, to_bytes=True)
        response = requests.post(
            url=str(self.api_url),
            data=msg_bytes,
        )

        if response.status_code != 200:
            raise requests.ConnectionError(
                f"Failed to fetch metadata. Response returned with code {response.status_code}"
            )

        result = _deserialize(response.content, from_bytes=True)
        return result


@serializable(recursive_serde=True)
class PythonConnection(NodeConnection):
    node: NewNode

    def __init__(self, node: NewNode) -> None:
        self.node = node

    def get_node_metadata(self) -> NodeMetadataJSON:
        return self.node.metadata().to(NodeMetadataJSON)

    def get_api(self, credentials: SyftSigningKey) -> SyftAPI:
        obj = self.node.get_api()
        obj.connection = self
        obj.signing_key = credentials
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
        if isinstance(result, OkErr):
            return result.value
        return result

    def connect(self, email: str, password: str) -> Optional[SyftSigningKey]:
        obj = self.exchange_credentials(email=email, password=password)
        if isinstance(obj, UserPrivateKey):
            return obj.signing_key
        print(obj)
        return None

    def make_call(self, signed_call: SyftAPICall) -> Union[Any, SyftError]:
        return self.node.handle_api_call(signed_call)


@instrument
class SyftClient:
    connection: NodeConnection
    metadata: Optional[NodeMetadataJSON]
    credentials: Optional[SyftSigningKey]

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
            self._fetch_node_metadata()

    @staticmethod
    def from_url(url: Union[str, GridURL]) -> Self:
        return SyftClient(connection=HTTPConnection(GridURL.from_url(url)))

    @staticmethod
    def from_node(node: NewNode) -> Self:
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
        if self._api is None:
            self._fetch_api(self.credentials)

        return self._api

    def upload_dataset(self, dataset: CreateDataset) -> Union[SyftSuccess, SyftError]:
        for asset in tqdm(dataset.asset_list):
            print(f"Uploading: {asset.name}")
            # response = asset.data.new_send(self)
            # if isinstance(response, SyftError):
            #     print(f"Failed to upload asset\n: {asset}")
            #     return response
            # data_ptr = response
            # asset.action_id = data_ptr.id
            asset.node_uid = self.id
        valid = dataset.check()
        if valid.ok():
            return self.api.services.dataset.add(dataset=dataset)
        else:
            if len(valid.err()) > 0:
                return tuple(valid.err())
            return valid.err()

    @property
    def data_subject_registry(self) -> Optional[APIModule]:
        if self.api is not None and hasattr(self.api.services, "data_subject"):
            return self.api.services.data_subject
        return None

    def connect(self, email: str, password: str, cache: bool = True) -> None:
        signing_key = self.connection.connect(email=email, password=password)
        if signing_key is not None:
            self.credentials = signing_key
            self._fetch_api(self.credentials)
            if cache:
                SyftClientSessionCache.add_client(
                    email=email,
                    password=password,
                    connection=self.connection,
                    syft_client=self,
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
        return f"<{type(self).__name__} - {self.name}: {self.id} {self.connection}>"

    def _fetch_node_metadata(self) -> None:
        metadata = self.connection.get_node_metadata()
        metadata.check_version(__version__)
        self.metadata = metadata

    def _fetch_api(self, credentials: SyftSigningKey):
        _api = self.connection.get_api(credentials=credentials)
        APIRegistry.set_api_for(node_uid=self.id, api=_api)
        self._api = _api


@instrument
def login(
    url: Union[str, GridURL] = DEFAULT_PYGRID_ADDRESS,
    node: Optional[NewNode] = None,
    port: Optional[int] = None,
    email: Optional[str] = None,
    password: Optional[str] = None,
    cache: bool = True,
) -> SyftClient:
    if node:
        connection = PythonConnection(node=node)
    else:
        url = GridURL.from_url(url)
        if isinstance(port, (int, str)):
            url.set_port(int(port))
        connection = HTTPConnection(url=url)

    login_credentials = UserLoginCredentials(email=email, password=password)

    _client = None
    if cache:
        _client = SyftClientSessionCache.get_client(
            login_credentials.email,
            login_credentials.password,
            connection=connection,
        )
        if _client:
            print(
                f"Using cached client for {_client.name} as <{login_credentials.email}>"
            )

    if _client is None:
        _client = SyftClient(connection=connection)
        _client.connect(
            email=login_credentials.email,
            password=login_credentials.password,
            cache=cache,
        )
        if _client.credentials:
            print(f"Logged into {_client.name} as <{login_credentials.email}>")

    return _client


class SyftClientSessionCache:
    __credentials_store__: Dict = {}
    __cache_key_format__ = "{email}-{password}-{connection}"

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

    @classmethod
    def get_client(
        cls, email: str, password: str, connection: NodeConnection
    ) -> Optional[SyftClient]:
        hash_key = cls._get_key(email, password, connection.get_cache_key())
        return cls.__credentials_store__.get(hash_key, None)
