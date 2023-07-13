# future
from __future__ import annotations

# stdlib
from typing import Optional
from typing import TYPE_CHECKING

# relative
from ..client.api import APIRegistry
from ..serde.serializable import serializable
from ..service.network.routes import NodeRouteType
from ..service.response import SyftError
from ..service.response import SyftSuccess
from ..types.syft_object import SYFT_OBJECT_VERSION_1
from ..types.syft_object import SyftObject
from ..types.uid import UID
from .api import APIModule
from .client import SyftClient
from .client import login

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

    __api_patched = False

    @property
    def code(self) -> Optional[APIModule]:
        if self.api.has_service("code"):
            res = self.api.services.code
            # the order is important here
            # its also important that patching only happens once
            if not self.__api_patched:
                self._request_code_execution = res.request_code_execution
                self.__api_patched = True
            res.request_code_execution = self.request_code_execution
            return res
        return None

    @property
    def requests(self) -> Optional[APIModule]:
        if self.api.has_service("request"):
            return self.api.services.request
        return None

    def connect_to_gateway(
        self,
        via_client: Optional[SyftClient] = None,
        url: Optional[str] = None,
        port: Optional[int] = None,
        handle: Optional["NodeHandle"] = None,  # noqa: F821
        **kwargs,
    ) -> None:
        if via_client is not None:
            client = via_client
        elif handle is not None:
            client = handle.client
        else:
            client = login(url=url, port=port, **kwargs)
            if isinstance(client, SyftError):
                return client

        res = self.exchange_route(client)
        if isinstance(res, SyftSuccess):
            return SyftSuccess(
                message=f"Connected {self.metadata.node_type} to {client.name} gateway"
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
            # We would need the verify key of the data scientist to be able to index the correct client
            # Since we do not want the data scientist to pass in the clients to the enclave client
            # from a UX perspecitve.
            # we will use the recent node id to find the correct client
            # assuming that it is the correct client
            # Warning: This could lead to inconsistent results, when we have multiple clients
            # in the same node pointing to the same node.
            # One way, by which we could solve this in the long term,
            # by forcing the user to pass only assets to the sy.ExactMatch,
            # by which we could extract the verify key of the data scientist
            # as each object comes with a verify key and node_uid
            # the asset object would contain the verify key of the data scientist.
            api = APIRegistry.get_by_recent_node_uid(k.node_id)
            if api is None:
                raise ValueError(f"could not find client for input {v}")
            else:
                apis += [api]

        for api in apis:
            api.services.code.request_code_execution(code=code)

        # we are using the real method here, see the .code property getter
        _ = self.code
        res = self._request_code_execution(code=code)

        return res
