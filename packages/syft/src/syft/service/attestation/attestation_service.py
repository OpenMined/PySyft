# stdlib
from collections.abc import Callable

# third party
import requests

# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...util.util import str_to_bool
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import service_method
from ..user.user_roles import GUEST_ROLE_LEVEL
from .attestation_constants import ATTESTATION_SERVICE_URL
from .attestation_constants import ATTEST_CPU_ENDPOINT
from .attestation_constants import ATTEST_GPU_ENDPOINT


@serializable()
class AttestationService(AbstractService):
    """This service is responsible for getting all sorts of attestations for any client."""

    def __init__(self, store: DocumentStore) -> None:
        self.store = store

    def perform_request(
        self, method: Callable, endpoint: str, raw: bool = False
    ) -> SyftSuccess | SyftError | str:
        try:
            response = method(f"{ATTESTATION_SERVICE_URL}{endpoint}")
            response.raise_for_status()
            message = response.json().get("result")
            raw_token = response.json().get("token")
            if raw:
                return raw_token
            elif str_to_bool(message):
                return SyftSuccess(message=message)
            else:
                return SyftError(message=message)
        except requests.HTTPError:
            return SyftError(message=f"{response.json()['detail']}")
        except requests.RequestException as e:
            return SyftError(message=f"Failed to perform request. {e}")

    @service_method(
        path="attestation.get_cpu_attestation",
        name="get_cpu_attestation",
        roles=GUEST_ROLE_LEVEL,
    )
    def get_cpu_attestation(
        self, context: AuthedServiceContext, raw_token: bool = False
    ) -> str | SyftError | SyftSuccess:
        return self.perform_request(requests.get, ATTEST_CPU_ENDPOINT, raw_token)

    @service_method(
        path="attestation.get_gpu_attestation",
        name="get_gpu_attestation",
        roles=GUEST_ROLE_LEVEL,
    )
    def get_gpu_attestation(
        self, context: AuthedServiceContext, raw_token: bool = False
    ) -> str | SyftError | SyftSuccess:
        return self.perform_request(requests.get, ATTEST_GPU_ENDPOINT, raw_token)
