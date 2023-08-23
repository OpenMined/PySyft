# third party

# stdlib
import os

# relative
from ...serde.lib_service_registry import CMPClass
from ...serde.lib_service_registry import CMPFunction
from ...serde.lib_service_registry import CMPTree
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...types.file import SyftFolder
from ...util.telemetry import instrument
from ..context import AuthedServiceContext
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import LibConfigRegistry
from ..service import register_lib_obj
from ..service import service_method

# import subprocess

# import subprocess


@instrument
@serializable()
class PythonLibService(AbstractService):
    def __init__(self, store: DocumentStore) -> None:
        self.store = store

    @service_method(path="python_lib.add_lib", name="add_lib")
    def add_lib(
        self, context: AuthedServiceContext, syft_folder: SyftFolder, cmp: CMPTree
    ):
        path = syft_folder.model_folder

        proc = os.system(f"pip install -e {str(path)} > /tmp/out.txt")

        cmp = cmp.build()
        lib_wrapper_service = context.node.get_service("LibWrapperService")

        for lib_obj in cmp.flatten():
            if isinstance(lib_obj, CMPFunction) or isinstance(lib_obj, CMPClass):
                register_lib_obj(lib_obj)
                # stdlib
                import sys

                print(
                    lib_obj.path, lib_obj.pre_hook, lib_obj.post_hook, file=sys.stderr
                )
                if lib_obj.pre_hook is not None:
                    lib_wrapper_service.set_wrapper(context, lib_obj.pre_hook)
                if lib_obj.post_hook is not None:
                    lib_wrapper_service.set_wrapper(context, lib_obj.post_hook)

        return SyftSuccess(message="Lib added succesfully:" + str(proc))

    @service_method(path="python_lib.show_lib", name="show_lib")
    def show_lib(self, context: AuthedServiceContext):
        return list(LibConfigRegistry.__service_config_registry__.keys())

    @service_method(path="python_lib.send_cmp", name="send_cmp")
    def send_cmp(self, context: AuthedServiceContext, cmp: CMPTree):
        # stdlib
        import sys

        print(cmp, file=sys.stderr)
        return SyftSuccess(message="CMP received succesfully!")