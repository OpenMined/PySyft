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
from ...types.file import SyftFolder
from ...serde.lib_permissions import ALL_EXECUTE
# import subprocess
from gevent import subprocess
# import subprocess
from gevent.select import select
import sys
import pip
import asyncio
import gevent
import os

@instrument
@serializable()
class PythonLibService(AbstractService):
    def __init__(self, store: DocumentStore) -> None:
        self.store = store
    
    @service_method(path="python_lib.add_lib", name="add_lib")
    def add_lib(self, context: AuthedServiceContext, syft_folder: SyftFolder, cmp: CMPTree):
        from gevent import monkey
        monkey.patch_all()
        path = syft_folder.model_folder

        proc = os.system(f"pip install {str(path)} > /tmp/out.txt")

        cmp = cmp.build()
        
        for lib_obj in cmp.flatten():
            if isinstance(lib_obj, CMPFunction) or isinstance(lib_obj, CMPClass):
                register_lib_obj(lib_obj)
        return SyftSuccess(message="Lib added succesfully:" + str(proc))
    
    @service_method(path="python_lib.show_lib", name="show_lib")
    def show_lib(self, context: AuthedServiceContext):
        return list(LibConfigRegistry.__service_config_registry__.keys())
    
    @service_method(path="python_lib.send_cmp", name="send_cmp")
    def send_cmp(self, context: AuthedServiceContext, cmp: CMPTree):
        import sys
        print(cmp, file=sys.stderr)
        return SyftSuccess(message="CMP received succesfully!")