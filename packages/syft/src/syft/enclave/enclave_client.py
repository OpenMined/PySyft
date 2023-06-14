# future
from __future__ import annotations

# stdlib
from typing import List
from typing import Optional
from typing import TYPE_CHECKING

# relative
from ..client.api import SyftAPI
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
        owners (List[SyftClient]): The list of Domain Owners the enclave belongs to
        url (str): Connection URL to communicate with the enclave.
    """

    owners: List[SyftClient]
    url: str
    syft_enclave_client: Optional[SyftClient] = None

    def register(
        self,
        name: str,
        email: str,
        password: str,
        institution: Optional[str] = None,
        website: Optional[str] = None,
    ):
        guest_client = login(url=self.url)
        return guest_client.register(
            name=name,
            email=email,
            password=password,
            institution=institution,
            website=website,
        )

    def login(
        self,
        email: str,
        password: str,
    ) -> None:
        self.syft_enclave_client = login(url=self.url, email=email, password=password)

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
        guest_client = login(url=self.url)
        guest_client.credentials = signing_key
        self.syft_enclave_client = guest_client

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

        for domain_client in self.owners:
            domain_client.api.services.code.request_code_execution(code=code)

        res = self.api.services.code.request_code_execution(code=code)

        return res

    def __repr__(self):
        data = dict(self.dict(exclude={"syft_enclave_client"}))
        return f"{self.__class__.__name__}({data})"


@serializable()
class AzureEnclaveMetadata(EnclaveMetadata):
    url: str


class AzureEnclaveClient(EnclaveClient):
    def _get_enclave_metadata(self) -> AzureEnclaveMetadata:
        return AzureEnclaveMetadata(url=self.url)

    @staticmethod
    def from_enclave_metadata(
        enclave_metadata: AzureEnclaveMetadata, signing_key: SyftSigningKey
    ) -> AzureEnclaveClient:
        # In the context of Domain Owners, who would like to only communicate with enclave, would not provide owners
        azure_encalve_client = AzureEnclaveClient(
            owners=[], url=enclave_metadata.url, signing_key=signing_key
        )

        azure_encalve_client.login_by_signing_key(signing_key=signing_key)

        return azure_encalve_client
