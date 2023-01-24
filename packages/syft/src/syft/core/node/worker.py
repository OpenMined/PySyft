# future
from __future__ import annotations

# stdlib
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
from ...core.node.common.node_table.syft_object import SyftObject
from ..common.serde.serializable import serializable
from ..common.uid import UID
from .new.action_service import ActionService
from .new.api import SignedSyftAPICall
from .new.api import SyftAPICall
from .new.credentials import SyftSigningKey
from .new.service import AbstractService
from .new.service import ServiceConfigRegistry
from .new.user import UserCollection

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


@serializable(recursive_serde=True)
class Worker:
    signing_key: Optional[SigningKey]
    required_signed_calls: bool = True

    def __init__(
        self,
        *,  # Trasterisk
        name: Optional[str] = None,
        id: Optional[UID] = UID(),
        services: Optional[List[Type[AbstractService]]] = None,
        signing_key: Optional[SigningKey] = SigningKey.generate(),
        user_collection: Optional[UserCollection] = None,
    ):
        # 🟡 TODO 22: change our ENV variable format and default init args to make this
        # less horrible or add some convenience functions
        if node_uid_env is not None:
            self.id = UID.from_string(node_uid_env)
        else:
            self.id = id

        if signing_key_env is not None:
            self.signing_key = SyftSigningKey(
                SigningKey(bytes.fromhex(signing_key_env))
            )
        else:
            self.signing_key = SyftSigningKey(signing_key)

        print("============> Starting Worker with:", self.id, self.signing_key)

        self.name = name
        services = [UserCollection, ActionService] if services is None else services
        self.services = services
        self.service_config = ServiceConfigRegistry.get_registered_configs()
        self._construct_services()
        self.post_init()

    def post_init(self) -> None:
        pass
        # super().post_init()

    def _construct_services(self):
        self.service_path_map = {}
        for service_klass in self.services:
            self.service_path_map[service_klass.__name__] = service_klass(self)

    def _get_service_method_from_path(self, path: str) -> Callable:

        path_list = path.split(".")
        method_name = path_list.pop()
        service_name = path_list.pop()
        service_obj = self.service_path_map[service_name]

        return getattr(service_obj, method_name)

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
                f"You sent a {type(api_call)}. This node requires SignedSyftAPICall."
            )
        else:
            if not api_call.is_valid:
                return Err("Your message signature is invalid")

        credentials = api_call.credentials
        api_call = api_call.message

        # 🔵 TODO 4: Add @service decorator to autobind services into the SyftAPI
        if api_call.path not in self.service_config:
            return Err(f"API call not in registered services: {api_call.path}")

        _private_api_path = ServiceConfigRegistry.private_path_for(api_call.path)

        method = self._get_service_method_from_path(_private_api_path)
        result = method(credentials, *api_call.args, **api_call.kwargs)
        return result

        # if api_call.path == "services.user.create":
        #     result = self.user_collection.create(
        #         credentials=credentials, **api_call.kwargs
        #     )
        #     return result
        # elif api_call.path == "services.action.set":
        #     result = self.action_service.set(credentials=credentials, **api_call.kwargs)
        #     return result
        # elif api_call.path == "services.action.get":
        #     result = self.action_service.get(credentials=credentials, **api_call.kwargs)
        #     return result
        # elif api_call.path == "services.action.execute":
        #     result = self.action_service.execute(
        #         credentials=credentials, **api_call.kwargs
        #     )
        #     return result
        # return Err("Wrong path")
