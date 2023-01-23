# stdlib
import hashlib
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
from ....core.common.serde.deserialize import _deserialize
from ....grid import GridURL
from ....logger import debug
from ....util import verify_tls
from ...common.uid import UID
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


class SyftClient:
    proxies: Dict[str, str] = {}
    url: GridURL
    node_name: str
    node_uid: UID
    credentials: Optional[SyftSigningKey]

    ROUTE_API_CALL = f"{API_PATH}/new_api_call"
    ROUTE_LOGIN = f"{API_PATH}/new_login"
    ROUTE_API = f"{API_PATH}/new_api"
    METADATA_API = "/api/v1/syft/metadata"
    _session: Optional[Session]

    def __init__(
        self,
        url: GridURL,
        node_name: str,
        node_uid: UID,
        credentials: Optional[SyftSigningKey] = None,
    ) -> None:
        self.url = url
        self.node_name = node_name
        self.node_uid = node_uid
        self.credentials: Optional[SyftSigningKey] = credentials
        self._api = None
        self._session = None

    @staticmethod
    def from_url(url: Union[str, GridURL]) -> Self:
        return SyftClient(
            url=GridURL.from_url(url), node_name="Anonymous", node_uid=UID()
        )

    @property
    def icon(self) -> str:
        return "📡"

    def __hash__(self) -> int:
        return hash(self.node_uid)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SyftClient):
            return False
        return (
            self.node_uid == other.node_uid
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
        return f"<{type(self).__name__} - {self.node_name}: {self.node_uid.no_dash}>"

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

    def _set_node_metadata(self) -> None:
        response = self._make_get(self.METADATA_API)
        metadata = _deserialize(response, from_bytes=True)
        self.node_name = metadata.name
        self.node_uid = metadata.id

    def _get_api(self) -> SyftAPI:
        content = self._make_get(self.ROUTE_API)
        obj = _deserialize(content, from_bytes=True)
        obj.api_url = self.url.with_path(self.ROUTE_API_CALL)
        return cast(SyftAPI, obj)

    # public attributes

    def _set_api(self):
        _api = self._get_api()
        APIRegistry.set_api_for(node_uid=self.node_uid, api=_api)
        self._api = _api

    @property
    def api(self) -> SyftAPI:
        if self._api is None:
            self._set_api()

        return self._api

    def login(self, email: str, password: str) -> None:
        credentials = {"email": email, "password": password}
        response = self._make_post(self.ROUTE_LOGIN, credentials)
        obj = _deserialize(response, from_bytes=True)
        self.credentials = obj.signing_key
        self._set_node_metadata()
        self._set_api()
        SyftClientSessionCache.add_user(
            email=email, password=password, syft_client=self
        )


class SyftClientSessionCache:
    __credentials_store__: Dict = {}
    __cache_key_format__ = "{email}-{password}"

    @classmethod
    def _get_key(cls, email: str, password: str) -> str:
        key = cls.__cache_key_format__.format(email=email, password=password)
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    @classmethod
    def add_user(cls, email: str, password: str, syft_client: SyftClient):
        hash_key = cls._get_key(email, password)
        cls.__credentials_store__[hash_key] = syft_client

    @classmethod
    def get_user_session(cls, email: str, password: str) -> Optional[SyftClient]:
        hash_key = cls._get_key(email, password)
        return cls.__credentials_store__.get(hash_key, None)
