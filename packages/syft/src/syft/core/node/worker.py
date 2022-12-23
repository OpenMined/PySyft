# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Optional
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
from .new.user import UserCollection


class TestObject(SyftObject):
    # version
    __canonical_name__ = "TestObject"
    __version__ = 1

    # fields
    name: str

    # serde / storage rules
    __attr_state__ = [
        "name",
    ]
    __attr_searchable__ = ["name"]
    __attr_unique__ = ["name"]


@serializable(recursive_serde=True)
class Worker:
    signing_key: Optional[SigningKey]
    required_signed_calls: bool = True

    def __init__(
        self,
        *,  # Trasterisk
        name: Optional[str] = None,
        id: Optional[UID] = None,
        signing_key: Optional[SigningKey] = None,
        action_service: Optional[ActionService] = None,
        user_collection: Optional[UserCollection] = None,
    ):
        if id is None:
            id = UID()
        self.id = id
        self.name = name
        self.signing_key = signing_key
        if action_service is None:
            action_service = ActionService(node_uid=self.id)
        self.action_service = action_service
        if user_collection is None:
            user_collection = UserCollection(node_uid=self.id)
        self.user_collection = user_collection
        self.post_init()

    def post_init(self) -> None:
        pass
        # super().post_init()

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

        # 🟡 TODO: Replace with the @service decorator binding

        if api_call.path == "services.user.create":
            result = self.user_collection.create(
                credentials=credentials, **api_call.kwargs
            )
            return result
        elif api_call.path == "services.action.set":
            result = self.action_service.set(credentials=credentials, **api_call.kwargs)
            return result
        elif api_call.path == "services.action.get":
            result = self.action_service.get(credentials=credentials, **api_call.kwargs)
            return result
        elif api_call.path == "services.action.execute":
            result = self.action_service.execute(
                credentials=credentials, **api_call.kwargs
            )
            return result

        return Err("Wrong path")
