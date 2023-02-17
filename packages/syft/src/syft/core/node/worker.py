# future
from __future__ import annotations

# stdlib
from functools import partial
import os
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Type
from typing import Union

# third party
from nacl.signing import SigningKey
from result import Err
from result import Result

# relative
from ... import __version__
from ...core.node.common.node_table.syft_object import HIGHEST_SYFT_OBJECT_VERSION
from ...core.node.common.node_table.syft_object import LOWEST_SYFT_OBJECT_VERSION
from ...core.node.common.node_table.syft_object import SyftObject
from ...telemetry import instrument
from ...util import random_name
from ..common.serde.serializable import serializable
from ..common.uid import UID
from .new.action_service import ActionService
from .new.action_store import ActionStore
from .new.api import SignedSyftAPICall
from .new.api import SyftAPI
from .new.api import SyftAPICall
from .new.context import AuthedServiceContext
from .new.context import NodeServiceContext
from .new.context import UnauthedServiceContext
from .new.context import UserLoginCredentials
from .new.credentials import SyftSigningKey
from .new.data_subject_service import DataSubjectService
from .new.dataset_service import DatasetService
from .new.dict_document_store import DictStoreConfig
from .new.document_store import StoreConfig
from .new.node import NewNode
from .new.node_metadata import NodeMetadata
from .new.request_service import RequestService
from .new.service import AbstractService
from .new.service import ServiceConfigRegistry
from .new.test_service import TestService
from .new.user import User
from .new.user import UserCreate
from .new.user_code_service import UserCodeService
from .new.user_service import UserService
from .new.user_stash import UserStash
from .new.policy_code_service import PolicyCodeService
from .new.policy_service import PolicyService


NODE_PRIVATE_KEY = "NODE_PRIVATE_KEY"
NODE_UID = "NODE_UID"


def get_private_key_env() -> Optional[str]:
    return get_env(NODE_PRIVATE_KEY)


def get_node_uid_env() -> Optional[str]:
    return get_env(NODE_UID)


def get_env(key: str) -> Optional[str]:
    return os.environ.get(key, None)


signing_key_env = get_private_key_env()
node_uid_env = get_node_uid_env()


@instrument
@serializable(recursive_serde=True)
class Worker(NewNode):
    signing_key: Optional[SyftSigningKey]
    required_signed_calls: bool = True

    def __init__(
        self,
        *,  # Trasterisk
        name: Optional[str] = None,
        id: Optional[UID] = None,
        services: Optional[List[Type[AbstractService]]] = None,
        signing_key: Optional[SigningKey] = SigningKey.generate(),
        store_config: Optional[StoreConfig] = None,
        root_email: str = "info@openmined.org",
        root_password: str = "changethis",
    ):
        # 🟡 TODO 22: change our ENV variable format and default init args to make this
        # less horrible or add some convenience functions
        if node_uid_env is not None:
            self.id = UID.from_string(node_uid_env)
        else:
            if id is None:
                id = UID()
            self.id = id

        if signing_key_env is not None:
            self.signing_key = SyftSigningKey.from_string(signing_key_env)
        else:
            self.signing_key = SyftSigningKey(signing_key=signing_key)

        if name is None:
            name = random_name()
        self.name = name
        services = (
            [
                UserService,
                ActionService,
                TestService,
                DatasetService,
                UserCodeService,
                RequestService,
                DataSubjectService,
                PolicyCodeService,
                PolicyService
            ]
            if services is None
            else services
        )
        self.services = services
        self.service_config = ServiceConfigRegistry.get_registered_configs()
        store_config = DictStoreConfig() if store_config is None else store_config
        self.init_stores(store_config=store_config)
        self._construct_services()
        create_admin_new(  # nosec B106
            name="Jane Doe",
            email="info@openmined.org",
            password="changethis",
            node=self,
        )
        self.post_init()

    def __repr__(self) -> str:
        return f"{type(self).__name__}: {self.name} - {self.id} {self.services}"

    def post_init(self) -> None:
        print(f"Starting {self}")
        # super().post_init()

    def init_stores(self, store_config: StoreConfig):
        document_store = store_config.store_type
        self.document_store = document_store(client_config=store_config.client_config)

    def _construct_services(self):
        self.service_path_map = {}
        for service_klass in self.services:
            kwargs = {}
            if service_klass == ActionService:
                action_store = ActionStore(root_verify_key=self.signing_key.verify_key)
                kwargs["store"] = action_store
            if service_klass in [
                UserService,
                DatasetService,
                UserCodeService,
                RequestService,
                DataSubjectService,
                PolicyCodeService,
                PolicyService
            ]:
                kwargs["store"] = self.document_store
            self.service_path_map[service_klass.__name__.lower()] = service_klass(
                **kwargs
            )

    def get_service_method(self, path_or_func: Union[str, Callable]) -> Callable:
        if callable(path_or_func):
            path_or_func = path_or_func.__qualname__
        return self._get_service_method_from_path(path_or_func)

    def get_service(self, path_or_func: Union[str, Callable]) -> Callable:
        if callable(path_or_func):
            path_or_func = path_or_func.__qualname__
        return self._get_service_from_path(path_or_func)

    def _get_service_from_path(self, path: str) -> AbstractService:
        path_list = path.split(".")
        if len(path_list) > 1:
            _ = path_list.pop()
        service_name = path_list.pop()
        return self.service_path_map[service_name.lower()]

    def _get_service_method_from_path(self, path: str) -> Callable:
        path_list = path.split(".")
        method_name = path_list.pop()
        service_obj = self._get_service_from_path(path=path)

        return getattr(service_obj, method_name)

    def metadata(self) -> NodeMetadata:
        return NodeMetadata(
            name=self.name,
            id=self.id,
            verify_key=self.signing_key.verify_key,
            highest_object_version=HIGHEST_SYFT_OBJECT_VERSION,
            lowest_object_version=LOWEST_SYFT_OBJECT_VERSION,
            syft_version=__version__,
        )

    @property
    def icon(self) -> str:
        return "🦾"

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Worker):
            return False

        if self.id != other.id:
            return False

        return True

    def handle_api_call(
        self, api_call: Union[SyftAPICall, SignedSyftAPICall]
    ) -> Result[SyftObject, Err]:
        if self.required_signed_calls and isinstance(api_call, SyftAPICall):
            return Err(
                f"You sent a {type(api_call)}. This node requires SignedSyftAPICall."  # type: ignore
            )
        else:
            if not api_call.is_valid:
                return Err("Your message signature is invalid")  # type: ignore

        credentials = api_call.credentials
        api_call = api_call.message

        context = AuthedServiceContext(node=self, credentials=credentials)

        # 🔵 TODO 4: Add @service decorator to autobind services into the SyftAPI
        if api_call.path not in self.service_config:
            return Err(f"API call not in registered services: {api_call.path}")  # type: ignore

        _private_api_path = ServiceConfigRegistry.private_path_for(api_call.path)

        method = self.get_service_method(_private_api_path)
        result = method(context, *api_call.args, **api_call.kwargs)
        return result

    def get_api(self) -> SyftAPI:
        return SyftAPI.for_user(node=self)

    def get_method_with_context(
        self, function: Callable, context: NodeServiceContext
    ) -> Callable:
        method = self.get_service_method(function)
        return partial(method, context=context)

    def get_unauthed_context(
        self, login_credentials: UserLoginCredentials
    ) -> NodeServiceContext:
        return UnauthedServiceContext(node=self, login_credentials=login_credentials)


def create_admin_new(
    name: str,
    email: str,
    password: str,
    node: NewNode,
) -> Optional[User]:
    try:
        user_stash = UserStash(store=node.document_store)
        row_exists = user_stash.get_by_email(email=email).ok()
        if row_exists:
            return None
        else:
            create_user = UserCreate(
                name=name, email=email, password=password, password_verify=password
            )
            # New User Initialization
            user = user_stash.set(user=create_user.to(User))
            return user.ok()
    except Exception as e:
        print("create_admin failed", e)
