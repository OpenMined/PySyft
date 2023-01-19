# stdlib
from typing import List
from typing import Type

# third party
from nacl.signing import SigningKey

# relative
from ...common.uid import UID
from ...io.route import Route
from .api import APIRegistry
from .api import SyftAPI


class SyftClient:
    credentials: SigningKey
    id: UID
    routes: List[Type[Route]]

    def __init__(
        self, node_uid: str, signing_key: str, routes: List[Type[Route]]
    ) -> None:
        self.id = UID.from_string(node_uid)
        self.credentials = SigningKey(bytes.fromhex(signing_key))
        self.routes = routes
        self._api = None

    @property
    def api(self) -> SyftAPI:
        if self._api is not None:
            return self._api
        _api = self.routes[0].connection._get_api()
        APIRegistry.set_api_for(node_uid=self.id, api=_api)
        self._api = _api
        return _api
