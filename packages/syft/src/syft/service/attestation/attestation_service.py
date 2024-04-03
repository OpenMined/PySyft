# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ..context import AuthedServiceContext
from ..service import AbstractService
from ..service import service_method
from ..user.user_roles import GUEST_ROLE_LEVEL


@serializable()
class AttestationService(AbstractService):
    """This service is responsible for getting all sorts of attestations for any client."""

    def __init__(self, store: DocumentStore) -> None:
        self.store = store

    @service_method(
        path="attestation.get_attestation",
        name="get_attestation",
        roles=GUEST_ROLE_LEVEL,
    )
    def get_attestation(self, context: AuthedServiceContext) -> str:
        return "Checking attestation end point"
