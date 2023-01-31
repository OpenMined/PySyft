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
from typing_extensions import Self

# relative
from .... import __version__
from ....core.common.serde.deserialize import _deserialize
from ....grid import GridURL
from ....logger import debug
from ....util import verify_tls
from ...common.uid import UID
from ...node.new.credentials import UserLoginCredentials
from ...node.new.node_metadata import NodeMetadataJSON
from .api import APIRegistry
from .api import SyftAPI
from .credentials import SyftSigningKey

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
DEFAULT_PYGRID_ADDRESS = f"http://127.0.0.1:{DEFAULT_PYGRID_PORT}"


class SyftClient:
    proxies: Dict[str, str] = {}
    url: GridURL
    metadata: Optional[NodeMetadataJSON]
    credentials: Optional[SyftSigningKey]
    routes: Routes = Routes
    _session: Optional[Session]

    def __init__(
        self,
        url: GridURL,
        metadata: Optional[NodeMetadataJSON] = None,
        credentials: Optional[SyftSigningKey] = None,
        api: Optional[SyftAPI] = None,
    ) -> None:
        self.url = url
        self.metadata = metadata
        self.credentials: Optional[SyftSigningKey] = credentials
        self._api = api
        self._session = None
        self.post_init()

    def post_init(self) -> None:
        if self.metadata is None:
            self._set_node_metadata()

    @staticmethod
    def from_url(url: Union[str, GridURL]) -> Self:
        return SyftClient(url=GridURL.from_url(url))

    @property
    def name(self) -> Optional[str]:
        return self.metadata.name if self.metadata else None

    @property
    def id(self) -> Optional[UID]:
        return self.metadata.id if self.metadata else None

    @property
    def icon(self) -> str:
        return "📡"

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SyftClient):
            return False
        return (
            self.metadata == other.metadata
            and self.url == other.url
            and self.credentials == other.credentials
        )

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

    def __repr__(self) -> str:
        return f"<{type(self).__name__} - {self.name}: {self.id}>"

    def _make_get(self, path: str) -> bytes:
        url = self.url.with_path(path)
        response = self.session.get(
            url, verify=verify_tls(), proxies=SyftClient.proxies
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
            url, verify=verify_tls(), json=json, proxies=SyftClient.proxies
        )
        if response.status_code != 200:
            raise requests.ConnectionError(
                f"Failed to fetch {url}. Response returned with code {response.status_code}"
            )

        # upgrade to tls if available
        self.url = upgrade_tls(self.url, response)

        return response.content

    def _get_node_metadata(self) -> NodeMetadataJSON:
        response = self._make_get(self.routes.ROUTE_METADATA.value)
        metadata_json = json.loads(response)
        return NodeMetadataJSON(**metadata_json)

    def _set_node_metadata(self) -> None:
        metadata = self._get_node_metadata()
        metadata.check_version(__version__)
        self.metadata = metadata

    def _get_api(self) -> SyftAPI:
        content = self._make_get(self.routes.ROUTE_API.value)
        obj = _deserialize(content, from_bytes=True)
        obj.api_url = self.url.with_path(self.routes.ROUTE_API_CALL.value)
        return cast(SyftAPI, obj)

    # public attributes

    def _set_api(self):
        _api = self._get_api()
        APIRegistry.set_api_for(node_uid=self.id, api=_api)
        self._api = _api

    @property
    def api(self) -> SyftAPI:
        if self._api is None:
            self._set_api()

        return self._api

    def login(self, email: str, password: str) -> None:
        credentials = {"email": email, "password": password}
        response = self._make_post(self.routes.ROUTE_LOGIN.value, credentials)
        obj = _deserialize(response, from_bytes=True)
        self.credentials = obj.signing_key
        self._set_api()
        SyftClientSessionCache.add_client(
            email=email, password=password, url=self.url, syft_client=self
        )


def connect(
    url: Union[str, GridURL] = DEFAULT_PYGRID_ADDRESS,
    port: Optional[int] = None,
    email: Optional[str] = None,
    password: Optional[str] = None,
) -> SyftClient:
    if isinstance(port, (int, str)):
        url = GridURL.from_url(url).set_port(int(port))

    login_credentials = UserLoginCredentials(email=email, password=password)

    _client = SyftClientSessionCache.get_client(
        login_credentials.email,
        login_credentials.password,
        url=str(GridURL.from_url(url)),
    )

    if _client is None:
        _client = SyftClient.from_url(url)
        _client.login(
            email=login_credentials.email, password=login_credentials.password
        )

    print(f"Logged into {_client.name} as <{login_credentials.email}>")

    return _client


class SyftClientSessionCache:
    __credentials_store__: Dict = {}
    __cache_key_format__ = "{email}-{password}-{url}"

    @classmethod
    def _get_key(cls, email: str, password: str, url: str) -> str:
        key = cls.__cache_key_format__.format(email=email, password=password, url=url)
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    @classmethod
    def add_client(cls, email: str, password: str, url: str, syft_client: SyftClient):
        hash_key = cls._get_key(email, password, url)
        cls.__credentials_store__[hash_key] = syft_client

    @classmethod
    def get_client(cls, email: str, password: str, url: str) -> Optional[SyftClient]:
        hash_key = cls._get_key(email, password, url)
        return cls.__credentials_store__.get(hash_key, None)
