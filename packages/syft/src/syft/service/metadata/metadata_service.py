# stdlib

# relative
from ...serde.serializable import serializable
from ...store.db.db import DBManager
from ..context import AuthedServiceContext
from ..service import AbstractService
from ..service import service_method
from ..user.user_roles import GUEST_ROLE_LEVEL
from .server_metadata import ServerMetadata


@serializable(canonical_name="MetadataService", version=1)
class MetadataService(AbstractService):
    def __init__(self, store: DBManager) -> None:
        pass

    @service_method(
        path="metadata.get_metadata", name="get_metadata", roles=GUEST_ROLE_LEVEL
    )
    def get_metadata(self, context: AuthedServiceContext) -> ServerMetadata:
        return context.server.metadata  # type: ignore

    @service_method(path="metadata.get_env", name="get_env", roles=GUEST_ROLE_LEVEL)
    def get_env(self, context: AuthedServiceContext) -> str:
        return context.server.packages
