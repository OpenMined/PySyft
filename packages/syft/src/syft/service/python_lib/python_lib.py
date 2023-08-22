# stdlib
import ast
from enum import Enum
import inspect
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
from result import Ok
from result import Result

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...util.telemetry import instrument
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import service_method
from ..user.user_roles import GUEST_ROLE_LEVEL

@serializable()
class LibWrapperOrder(Enum):
    PRE_HOOK = "pre_hook"
    POST_HOOK = "post_hook"

@serializable()
class LibWrapper(SyftObject):
    # version
    __canonical_name__ = "LibWrapper"
    __version__ = SYFT_OBJECT_VERSION_1

    path: str
    order: LibWrapperOrder
    wrapper_code: str
    func_name: str

    __attr_searchable__ = ["path", "order"]
    __attr_unique__ = []

    def exec(self, context: AuthedServiceContext, arg: Any) -> Any:
        try:
            inner_function = ast.parse(self.wrapper_code).body[0]
            inner_function.decorator_list = []
            # compile the function
            raw_byte_code = compile(ast.unparse(inner_function), "<string>", "exec")
            # load it
            exec(raw_byte_code)  # nosec
            # execute it
            evil_string = f"{self.func_name}(context, arg)"
            result = eval(evil_string, None, locals())  # nosec
            # return the results
            return context, result
        except Exception as e:
            print(f"Failed to run LibWrapper Code. {e}")


def lib_pre_hook(path: str) -> LibWrapper:
    return lib_wrapper(path=path, order=LibWrapperOrder.PRE_HOOK)


def lib_post_hook(path: str) -> LibWrapper:
    return lib_wrapper(path=path, order=LibWrapperOrder.POST_HOOK)


def lib_wrapper(path: str, order: LibWrapperOrder) -> LibWrapper:
    def decorator(f):
        res = LibWrapper(
            path=path,
            order=order,
            wrapper_code=inspect.getsource(f),
            func_name=f.__name__,
        )
        return res

    return decorator

@serializable()
class LibWrapperStash(BaseUIDStoreStash):
    object_type = LibWrapper
    settings: PartitionSettings = PartitionSettings(
        name=LibWrapper.__canonical_name__, object_type=LibWrapper
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def get_by_path(
        self, credentials: SyftVerifyKey, path: str
    ) -> Result[List[LibWrapper], str]:
        # qks = QueryKeys(qks=[PathPartitionKey.with_obj(path)])
        results = self.get_all(credentials=credentials)
        items = []
        if results.is_ok() and results.ok():
            results = results.ok()
            for result in results:
                if result.path == path:
                    items.append(result)
            return Ok(items)
        else:
            return results
        # TODO: fix ability to query on a single index like path
        # return self.query_all(credentials, qks=qks)

    def update(
        self, credentials: SyftVerifyKey, wrapper: LibWrapper
    ) -> Result[LibWrapper, str]:
        res = self.check_type(wrapper, LibWrapper)
        if res.is_err():
            return res
        result = super().set(
            credentials=credentials, obj=res.ok(), ignore_duplicates=True
        )
        return result

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
