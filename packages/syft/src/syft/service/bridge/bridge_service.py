# stdlib
from typing import Any
from typing import List
from typing import Optional
from typing import Union

# third party
from result import Result

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...util.telemetry import instrument
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import service_method
from ..user.user_roles import GUEST_ROLE_LEVEL
from .api_bridge import APIBridge

TitlePartitionKey = PartitionKey(key="name", type_=str)


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


@instrument
@serializable()
class BridgeService(AbstractService):
    store: DocumentStore
    stash: BridgeStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = BridgeStash(store=store)

    @service_method(path="bridge.add", name="add")
    def add(
        self, context: AuthedServiceContext, url: str, base_url: Optional[str] = None
    ) -> Union[SyftSuccess, SyftError]:
        """Register a an API Bridge."""

        bridge = APIBridge.from_url(url, base_url)
        result = self.stash.add(context.credentials, bridge=bridge)
        if result.is_ok():
            return SyftSuccess(message=f"API Bridge added: {url}")
        return SyftError(message=f"Failed to add API Bridge {url}. {result.err()}")

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

        results = self.stash.get_all(context.credentials)
        if not results.is_ok():
            return SyftError(message=f"Bridge: {bridge} does not exist")
        results = results.ok()
        bridge = results[0]
        method = bridge.openapi._operation_map[method_name]
        callable_op = bridge.openapi._get_callable(method.request)
        parameters = {}
        data = {}

        for param in method.parameters:
            name = param.name
            if name in kwargs:
                parameters[name] = kwargs[name]

        if method.requestBody:
            if "application/json" in method.requestBody.content:
                schema = method.requestBody.content["application/json"].schema._proxy
                title = schema.title.lower()
                if title in kwargs:
                    data = kwargs[title]

        result = callable_op(parameters=parameters, data=data)
        print("got result", result)
        return result
