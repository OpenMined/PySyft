import gevent.subprocess as subprocess
from ...serde.serializable import serializable
from ...util.telemetry import instrument
from ..service import AbstractService
from ..service import service_method
from ..context import AuthedServiceContext
from ..user.user_roles import GUEST_ROLE_LEVEL
from ..response import SyftError
from ..response import SyftSuccess
from ...store.document_store import DocumentStore

@instrument
@serializable()
class MetadataService(AbstractService):

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        
    @service_method(path="metadata.get_env", name="get_env", roles=GUEST_ROLE_LEVEL)
    def get_env(self, context: AuthedServiceContext):
        
        res = subprocess.getoutput(
            "pip list --format=freeze",
        )
        import sys
        print(res, file=sys.stderr)
        return SyftSuccess(message=res.stdout.decode())