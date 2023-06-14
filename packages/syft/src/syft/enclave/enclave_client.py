# stdlib
from typing import List
from typing import Optional

# relative
from ..client.api import SyftAPI
from ..client.client import SyftClient
from ..client.client import login
from ..types.base import SyftBaseModel


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


class AzureEnclaveClient(EnclaveClient):
    pass
