# stdlib
from typing import Optional

# third party
from typing_extensions import Self

# relative
from ..serde.serializable import serializable
from .api import APIModule
from .client import SyftClient


@serializable()
class EnclaveClient(SyftClient):
    # TODO: add widget repr for enclave client

    @property
    def code(self) -> Optional[APIModule]:
        if self.api is not None and self.api.has_service("code"):
            return self.api.services.code

    @property
    def requests(self) -> Optional[APIModule]:
        if self.api is not None and self.api.has_service("request"):
            return self.api.services.request
        return None

    def apply_to_gateway(self, client: Self) -> None:
        return self.exchange_route(client)
