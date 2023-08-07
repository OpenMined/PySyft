# stdlib
import ast
from enum import Enum
import inspect
from typing import Any
from typing import List
from typing import Optional
from typing import Union

# third party
from openapi3.errors import UnexpectedResponseError
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
from .api_bridge import APIBridge

TitlePartitionKey = PartitionKey(key="name", type_=str)
PathPartitionKey = PartitionKey(key="path", type_=str)


@instrument
@serializable()
class BridgeStash(BaseUIDStoreStash):
    object_type = APIBridge
    settings: PartitionSettings = PartitionSettings(
        name=APIBridge.__canonical_name__, object_type=APIBridge
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def get_by_name(
        self, credentials: SyftVerifyKey, name: str
    ) -> Result[Optional[APIBridge], str]:
        qks = QueryKeys(qks=[TitlePartitionKey.with_obj(name)])
        return self.query_one(credentials, qks=qks)

    def add(
        self, credentials: SyftVerifyKey, bridge: APIBridge
    ) -> Result[APIBridge, str]:
        res = self.check_type(bridge, APIBridge)
        # we dont use and_then logic here as it is hard because of the order of the arguments
        if res.is_err():
            return res
        return super().set(credentials=credentials, obj=res.ok())


@serializable()
class APIWrapperOrder(Enum):
    PRE_HOOK = "pre_hook"
    POST_HOOK = "post_hook"


@serializable()
class APIWrapper(SyftObject):
    # version
    __canonical_name__ = "APIWrapper"
    __version__ = SYFT_OBJECT_VERSION_1

    path: str
    order: APIWrapperOrder
    wrapper_code: str
    func_name: str

    __attr_searchable__ = ["path", "order"]
    __attr_unique__ = ["path", "order"]

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
            print(f"Failed to run APIWrapper Code. {e}")


def api_pre_hook(path: str) -> APIWrapper:
    return api_wrapper(path=path, order=APIWrapperOrder.PRE_HOOK)


def api_post_hook(path: str) -> APIWrapper:
    return api_wrapper(path=path, order=APIWrapperOrder.POST_HOOK)


def api_wrapper(path: str, order: APIWrapperOrder) -> APIWrapper:
    def decorator(f):
        res = APIWrapper(
            path=path,
            order=order,
            wrapper_code=inspect.getsource(f),
            func_name=f.__name__,
        )
        return res

    return decorator


@serializable()
class APIWrapperStash(BaseUIDStoreStash):
    object_type = APIWrapper
    settings: PartitionSettings = PartitionSettings(
        name=APIWrapper.__canonical_name__, object_type=APIWrapper
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def get_by_path(
        self, credentials: SyftVerifyKey, path: str
    ) -> Result[List[APIWrapper], str]:
        qks = QueryKeys(qks=[PathPartitionKey.with_obj(path)])
        return self.query_all(credentials, qks=qks)

    def update(
        self, credentials: SyftVerifyKey, wrapper: APIWrapper
    ) -> Result[APIWrapper, str]:
        res = self.check_type(wrapper, APIWrapper)
        if res.is_err():
            return res
        result = super().set(
            credentials=credentials, obj=res.ok(), ignore_duplicates=True
        )
        return result


@serializable()
class BridgeAdded(SyftSuccess):
    pass


@instrument
@serializable()
class BridgeService(AbstractService):
    store: DocumentStore
    stash: BridgeStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = BridgeStash(store=store)
        self.wrapper_stash = APIWrapperStash(store=store)
        self.fake_auth_result = None

    @service_method(path="bridge.add", name="add")
    def add(
        self, context: AuthedServiceContext, url: str, base_url: Optional[str] = None
    ) -> Union[BridgeAdded, SyftError]:
        """Register a an API Bridge."""

        bridge = APIBridge.from_url(url, base_url)
        result = self.stash.add(context.credentials, bridge=bridge)
        if result.is_ok():
            bridge.register_serde_types()
            return BridgeAdded(message=f"API Bridge added: {url}")
        return SyftError(message=f"Failed to add API Bridge {url}. {result.err()}")

    @service_method(path="bridge.set_wrapper", name="set_wrapper")
    def set_wrapper(
        self, context: AuthedServiceContext, wrapper: APIWrapper
    ) -> Union[SyftSuccess, SyftError]:
        """Register an APIWrapper."""
        result = self.wrapper_stash.update(context.credentials, wrapper=wrapper)
        if result.is_ok():
            return SyftSuccess(message=f"APIWrapper added: {wrapper}")
        return SyftError(message=f"Failed to add APIWrapper {wrapper}. {result.err()}")

    @service_method(path="bridge.get_wrappers", name="get_wrappers")
    def get_wrappers(
        self, context: AuthedServiceContext, path: str
    ) -> Union[SyftSuccess, SyftError]:
        """Get an APIWrappers."""
        results = self.wrapper_stash.get_by_path(context.credentials, path=path)
        if results.is_ok() and results.ok():
            return results
        return SyftError(
            message=f"Failed to get APIWrapper for {path}. {results.err()}"
        )

    def get_all(
        self, context: AuthedServiceContext
    ) -> Union[List[APIBridge], SyftError]:
        results = self.stash.get_all(context.credentials)
        if results.is_ok():
            return results.ok()
        return SyftError(messages="Unable to get API Bridges")

    @service_method(path="bridge.call", name="call", roles=GUEST_ROLE_LEVEL)
    def call(
        self,
        context: AuthedServiceContext,
        bridge: str,
        method_name: str,
        **kwargs: Any,
    ) -> Union[SyftSuccess, SyftError]:
        """Call a Bridge API Method"""
        path = f"{bridge}.{method_name}"
        results = self.stash.get_all(context.node.verify_key)
        if not results.is_ok():
            return SyftError(message=f"Bridge: {bridge} does not exist")
        results = results.ok()
        bridge = results[0]

        if "HTTPBearer" in bridge.openapi.components.securitySchemes:
            token = context.session.get_auth(key="HTTPBearer")
            if token:
                bridge.openapi.authenticate("HTTPBearer", token)

        # if "OAuth2PasswordBearer" in bridge.openapi.components.securitySchemes:
        #     token = context.session.get_auth(key="OAuth2PasswordBearer")
        #     if token:
        #         bridge.openapi.authenticate("OAuth2PasswordBearer", token)

        method = bridge.openapi._operation_map[method_name]
        callable_op = bridge.openapi._get_callable(method.request)

        wrappers = self.wrapper_stash.get_by_path(context.node.verify_key, path=path)

        pre_wrapper = None
        post_wrapper = None
        if wrappers.is_ok() and wrappers.ok():
            wrappers = wrappers.ok()
            for wrapper in wrappers:
                if wrapper.order == APIWrapperOrder.PRE_HOOK:
                    pre_wrapper = wrapper
                elif wrapper.order == APIWrapperOrder.POST_HOOK:
                    post_wrapper = wrapper

        if pre_wrapper:
            context, kwargs = pre_wrapper.exec(context, kwargs)
            context.session.update_user_session()

        parameters, data = bridge.kwargs_to_parameters(method, kwargs)

        if self.fake_auth_result is not None:
            headers = {"Authorization": f"Bearer {self.fake_auth_result.access_token}"}
            # NOTE: we rely on workaround in openapi3 paths.py
            # to apply the session headers properly to the request
            callable_op.session.headers.update(headers)
        try:
            result = callable_op(parameters=parameters, data=data)
        except UnexpectedResponseError as e:
            if e.status_code in [401, 403]:
                return SyftError(
                    message=f"{e.status_code} Unauthorized. Call authenticate(token=)."
                )
        if post_wrapper:
            context, result = post_wrapper.exec(context, result)
        return result

    @service_method(
        path="bridge.authenticate", name="authenticate", roles=GUEST_ROLE_LEVEL
    )
    def authenticate(
        self, context: AuthedServiceContext, token: str
    ) -> Union[SyftSuccess, SyftError]:
        """Authenticate to Bridge API"""
        # let the user set a token to their session
        context.session.set_auth(key="HTTPBearer", value=token)
        return SyftSuccess(message="Token set.")

    @service_method(path="bridge.session", name="session", roles=GUEST_ROLE_LEVEL)
    def session(self, context: AuthedServiceContext) -> Union[SyftSuccess, SyftError]:
        """Show UserSession for Debugging"""
        return context.session
