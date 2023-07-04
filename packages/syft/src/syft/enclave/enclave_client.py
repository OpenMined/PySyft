# future
from __future__ import annotations

# stdlib
from typing import Optional
from typing import TYPE_CHECKING

# relative
from ..client.api import APIModule
from ..client.api import APIRegistry
from ..client.api import SyftAPI
from ..client.client import PythonConnection
from ..client.client import SyftClient
from ..client.client import login
from ..node.credentials import SyftSigningKey
from ..serde.serializable import serializable
from ..types.base import SyftBaseModel
from ..types.uid import UID
from .metadata import EnclaveMetadata

if TYPE_CHECKING:
    # relative
    from ..core.node.new.user_code import SubmitUserCode


class EnclaveClient(SyftBaseModel):
    """Base EnclaveClient class
    Args:
        domains (List[SyftClient]): List of domain nodes
        url (str): Connection URL to communicate with the enclave.
    """

    url: Optional[str] = None
    port: Optional[str] = None
    syft_enclave_client: Optional[SyftClient] = None

    def register(
        self,
        name: str,
        email: str,
        password: str,
        institution: Optional[str] = None,
        website: Optional[str] = None,
    ):
        if self.syft_enclave_client is not None:
            return self.syft_enclave_client.register(
                name, email, password, institution, website
            )
        else:
            guest_client = login(url=self.url)
            return guest_client.register(
                name=name,
                email=email,
                password=password,
                institution=institution,
                website=website,
            )

    def login(
        self, email: str, password: str, name: Optional[str] = None, register=False
    ) -> None:
        if register:
            self.register(name, email, password)
        if self.syft_enclave_client is not None:
            self.syft_enclave_client.login(email, password)
        else:
            self.syft_enclave_client = login(
                url=self.url, email=email, password=password
            )
        return self

    @property
    def settings(self):
        if not self.syft_enclave_client:
            raise Exception("Kindly login or register with the enclave")
        return self.syft_enclave_client.settings

    @property
    def api(self) -> SyftAPI:
        if not self.syft_enclave_client:
            raise Exception("Kindly login or register with the enclave")

        return self.syft_enclave_client.api

    def _get_enclave_metadata(self) -> EnclaveMetadata:
        raise NotImplementedError(
            "EnclaveClient subclasses must implement _get_enclave_metadata()"
        )

    def login_by_signing_key(self, signing_key: SyftSigningKey) -> None:
        if self.syft_enclave_client is None:
            guest_client = login(url=self.url)
            guest_client.credentials = signing_key
            self.syft_enclave_client = guest_client
        else:
            self.syft_enclave_client.credentials = signing_key

    def request_code_execution(self, code: SubmitUserCode):
        # relative
        from ..service.code.user_code_service import SubmitUserCode

        if not isinstance(code, SubmitUserCode):
            raise Exception(
                f"The input code should be of type: {SubmitUserCode} got:{type(code)}"
            )

        enclave_metadata = self._get_enclave_metadata()

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

    @property
    def code(self) -> Optional[APIModule]:
        if (
            self.syft_enclave_client is not None
            and self.syft_enclave_client.api.has_service("code")
        ):
            return self.syft_enclave_client.api.services.code

    def __repr__(self):
        data = dict(self.dict(exclude={"syft_enclave_client"}))
        return f"{self.__class__.__name__}({data})"


@serializable()
class AzureEnclaveMetadata(EnclaveMetadata):
    url: Optional[str]
    port: Optional[str]
    worker_id: Optional[UID]


class AzureEnclaveClient(EnclaveClient):
    def _get_enclave_metadata(self) -> AzureEnclaveMetadata:
        worker_id = None
        if self.syft_enclave_client is not None and isinstance(
            self.syft_enclave_client.connection, PythonConnection
        ):
            worker_id = self.syft_enclave_client.connection.node.id
        return AzureEnclaveMetadata(url=self.url, port=self.port, worker_id=worker_id)

    @staticmethod
    def from_enclave_metadata(
        enclave_metadata: AzureEnclaveMetadata, signing_key: SyftSigningKey
    ) -> AzureEnclaveClient:
        # In the context of Domain Owners, who would like to only communicate with enclave,
        #  would not provide domains to the enclave client.
        syft_enclave_client = None

        # python connection

        if enclave_metadata.worker_id is not None:
            # relative
            from ..node.node import NodeRegistry

            worker = NodeRegistry.node_for(enclave_metadata.worker_id)
            syft_enclave_client = worker.guest_client
        else:
            # syft absolute
            import syft as sy

            syft_enclave_client = sy.login(
                url=enclave_metadata.url, port=enclave_metadata.port
            )  # type: ignore
        # import ipdb
        # ipdb.set_trace()

        azure_enclave_client = AzureEnclaveClient(
            url=enclave_metadata.url,
            port=enclave_metadata.port,
            syft_enclave_client=syft_enclave_client,
        )

        azure_enclave_client.login_by_signing_key(signing_key=signing_key)

        return azure_enclave_client
