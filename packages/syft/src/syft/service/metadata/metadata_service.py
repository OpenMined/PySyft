# third party

# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...util.telemetry import instrument
from ..context import AuthedServiceContext
from ..service import AbstractService
from ..service import service_method
from ..user.user_roles import GUEST_ROLE_LEVEL


@instrument
@serializable()
class MetadataService(AbstractService):
    def __init__(self, store: DocumentStore) -> None:
        self.store = store

    @service_method(
        path="metadata.get_metadata", name="get_metadata", roles=GUEST_ROLE_LEVEL
    )
    def get_metadata(self, context: AuthedServiceContext):
        return context.node.metadata

    # @service_method(path="metadata.get_admin", name="get_admin", roles=GUEST_ROLE_LEVEL)
    # def get_admin(self, context: AuthedServiceContext):
    #     user_service = context.node.get_service("userservice")
    #     admin_user = user_service.get_all(context=context)[0]
    #     return admin_user

    @service_method(path="metadata.get_env", name="get_env", roles=GUEST_ROLE_LEVEL)
    def get_env(self, context: AuthedServiceContext):
        return context.node.packages
