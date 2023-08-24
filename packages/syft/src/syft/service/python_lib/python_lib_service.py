# third party

# stdlib
import os
from typing import Optional
from typing import Tuple
from typing import Union


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
from .python_lib import LibWrapperStash, LibWrapper, LibWrapperOrder
from ..response import SyftError
from ..response import SyftSuccess

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

@instrument
@serializable()
class LibWrapperService(AbstractService):
    store: DocumentStore
    stash: LibWrapperStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = LibWrapperStash(store=store)

    @service_method(path="lib_wrapper.set_wrapper", name="set_wrapper")
    def set_wrapper(
        self, context: AuthedServiceContext, wrapper: LibWrapper
    ) -> Union[SyftSuccess, SyftError]:
        """Register an APIWrapper."""
        result = self.stash.update(context.credentials, wrapper=wrapper)
        if result.is_ok():
            return SyftSuccess(message=f"APIWrapper added: {wrapper}")
        return SyftError(message=f"Failed to add APIWrapper {wrapper}. {result.err()}")

    @service_method(path="lib_wrapper.get_wrappers", name="get_wrappers")
    def get_wrappers(
        self, context: AuthedServiceContext, path: str
    ) -> Tuple[Optional[LibWrapper], Optional[LibWrapper]]:
        wrappers = self.stash.get_by_path(context.node.verify_key, path=path)
        pre_wrapper = None
        post_wrapper = None
        if wrappers.is_ok() and wrappers.ok():
            wrappers = wrappers.ok()
            for wrapper in wrappers:
                if wrapper.order == LibWrapperOrder.PRE_HOOK:
                    pre_wrapper = wrapper
                elif wrapper.order == LibWrapperOrder.POST_HOOK:
                    post_wrapper = wrapper

        return (pre_wrapper, post_wrapper)
