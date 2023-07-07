# future
from __future__ import annotations

# stdlib
from typing import Optional
from typing import TYPE_CHECKING

# third party
from typing_extensions import Self

# relative
from ..client.api import APIRegistry
from ..serde.serializable import serializable
from ..service.network.routes import NodeRouteType
from ..service.response import SyftSuccess
from ..types.syft_object import SYFT_OBJECT_VERSION_1
from ..types.syft_object import SyftObject
from ..types.uid import UID
from .api import APIModule
from .client import SyftClient

if TYPE_CHECKING:
    # relative
    from ..service.code.user_code import SubmitUserCode


@serializable()
class EnclaveMetadata(SyftObject):
    __canonical_name__ = "EnclaveMetadata"
    __version__ = SYFT_OBJECT_VERSION_1

    route: NodeRouteType


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
        res = self.exchange_route(client)
        if isinstance(res, SyftSuccess):
            return SyftSuccess(
                message=f"Connected {self.metadata.node_type} to gateway"
            )
        return res

    def get_enclave_metadata(self) -> EnclaveMetadata:
        return EnclaveMetadata(route=self.connection.route)

    def request_code_execution(self, code: SubmitUserCode):
        # relative
        from ..service.code.user_code_service import SubmitUserCode

        if not isinstance(code, SubmitUserCode):
            raise Exception(
                f"The input code should be of type: {SubmitUserCode} got:{type(code)}"
            )

        enclave_metadata = self.get_enclave_metadata()

        code_id = UID()
        code.id = code_id
        code.enclave_metadata = enclave_metadata

        apis = []
        for k, v in code.input_policy_init_kwargs.items():
            api = APIRegistry.api_for(k.node_id, k.verify_key)
            if api is None:
                raise ValueError(f"could not find client for input {v}")
            else:
                apis += [api]

        for api in apis:
            api.services.code.request_code_execution(code=code)

        res = self.api.services.code.request_code_execution(code=code)

        return res
