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


class SyftClientCache:
    __credentials_store__: Dict = {}
    __cache_key_format__ = "{username}-{password}"

    def _get_key(self, username: str, password: str) -> str:
        key = self.__cache_key_format__.format(username=username, password=password)
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def add_user(self, username: str, password: str, verify_key: str):
        hash_key = self._get_key(username, password)
        self.__credentials_store__[hash_key] = verify_key

    def get_user_key(self, username: str, password: str) -> Optional[SyftSigningKey]:
        hash_key = self._get_key(username, password)
        return self.__credentials_store__.get(hash_key, None)


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
    _session: Session

    def __init__(
        self,
        url: GridURL,
        node_name: str,
        node_uid: str,
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
        return SyftClient(url=GridURL.from_url(url), node_name="", node_uid="")

    @property
    def icon(self) -> str:
        return "📡"

    def __hash__(self) -> int:
        return hash(self.id)

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
        return f"<{type(self).__name__} - {self.name}: {self.node_uid.no_dash}>"

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

    def _get_api(self) -> SyftAPI:
        content = self._make_get(self.ROUTE_API)
        obj = _deserialize(content, from_bytes=True)
        obj.api_url = self.url.with_path(self.ROUTE_API_CALL)
        return cast(SyftAPI, obj)

    # public attributes

    @property
    def api(self) -> SyftAPI:
        if self._api is not None:
            return self._api
        _api = self._get_api()
        APIRegistry.set_api_for(node_uid=self.node_uid, api=_api)
        self._api = _api
        return _api

    def login(self, email: str, password: str) -> Dict:
        credentials = {"email": email, "password": password}
        response = self._make_post(self.ROUTE_LOGIN, credentials)
        obj = _deserialize(response, from_bytes=True)
        self.credentials = obj.signing_key
        self._get_api()
