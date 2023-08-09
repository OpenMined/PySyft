# third party

# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...util.telemetry import instrument
from ..context import AuthedServiceContext
from ..service import AbstractService
from ..service import service_method
from ..service import register_lib_obj
from ..user.user_roles import GUEST_ROLE_LEVEL
from ..response import SyftSuccess, SyftError
from ...serde.lib_service_registry import CMPModule, CMPFunction, CMPClass, CMPTree
from ..service import LibConfigRegistry

@instrument
@serializable()
class PythonLibService(AbstractService):
    def __init__(self, store: DocumentStore) -> None:
        self.store = store
    
    @service_method(path="python_lib.add_lib", name="add_lib")
    def add_lib(self, context: AuthedServiceContext, cmp_module: CMPTree):
        for lib_obj in cmp_module.flatten():
            if isinstance(lib_obj, CMPFunction) or isinstance(lib_obj, CMPClass):
                register_lib_obj(lib_obj)
        return SyftSuccess(message="Lib added succesfully!")
    
    @service_method(path="python_lib.show_lib", name="show_lib")
    def show_lib(self, context: AuthedServiceContext):
        return list(LibConfigRegistry.__service_config_registry__.keys())