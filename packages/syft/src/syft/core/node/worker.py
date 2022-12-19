# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Optional

# third party
from nacl.signing import SigningKey

# relative
from ...core.node.common.node_table.syft_object import SyftObject
from ..common.serde.serializable import serializable
from ..common.uid import UID
from .new.action_store import ActionStore
from .new.service_store import ServiceStore


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

    def __init__(
        self,
        *,  # Trasterisk
        name: Optional[str] = None,
        signing_key: Optional[SigningKey] = None,
        action_store: Optional[ActionStore] = None,
        service_store: Optional[ServiceStore] = None,
    ):
        self.id = UID()
        self.signing_key = signing_key
        self.action_store = action_store
        self.service_store = service_store
        self.post_init()

    def post_init(self) -> None:
        pass
        # super().post_init()

    @property
    def icon(self) -> str:
        return "🦾"

    @property
    def id(self) -> UID:
        return self.id

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Worker):
            return False

        if self.id != other.id:
            return False

        return True
