# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type

# relative
from ...common.uid import UID
from ...io.route import Route
from .api import APIRegistry
from .api import SyftAPI
from .credentials import SyftVerifyKey


class SyftClientCache:
    __credentials_store__: Dict = {}
    __cache_key_format__ = "{username}-{password}"

    def _get_key(self, username: str, password: str) -> int:
        key = self.__cache_key_format__.format(username=username, password=password)
        return hash(key)

    def add_user(self, username: str, password: str, verify_key: str):
        hash_key = self._get_key(username, password)
        self.__credentials_store__[hash_key] = verify_key

    def get_user_key(self, username: str, password: str):
        hash_key = self._get_key(username, password)
        return self.__credentials_store__.get(hash_key, None)


class SyftClient:
    name: str
    credentials: SyftVerifyKey
    id: UID
    routes: List[Type[Route]]
    access_token: str

    def __init__(
        self,
        node_name: str,
        node_uid: str,
        verify_key: str,
        routes: List[Type[Route]],
        access_token: Optional[str] = None,
    ) -> None:
        self.name = node_name
        self.id = UID.from_string(node_uid)
        self.credentials = SyftVerifyKey.from_string(verify_key)
        self.routes = routes
        self._api = None
        self.access_token = access_token

    @property
    def api(self) -> SyftAPI:
        if self._api is not None:
            return self._api
        _api = self.routes[0].connection._get_api()  # type: ignore
        APIRegistry.set_api_for(node_uid=self.id, api=_api)
        self._api = _api
        return _api

    @property
    def icon(self) -> str:
        return "🏰"

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SyftClient):
            return False

        if self.id != other.id:
            return False

        return True

    def __repr__(self) -> str:
        no_dash = str(self.id).replace("-", "")
        return f"<{type(self).__name__} - {self.name}: {no_dash}>"
