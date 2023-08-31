# future
from __future__ import annotations

# stdlib
from typing import TYPE_CHECKING

# third party
from typing_extensions import Self

# relative
from ..node.credentials import SyftVerifyKey
from ..serde.serializable import serializable
from .base import SyftBaseModel
from .uid import UID

if TYPE_CHECKING:
    # relative
    from ..client.client import SyftClient


class Identity(SyftBaseModel):
    node_id: UID
    verify_key: SyftVerifyKey

    __repr_attrs__ = ["id", "verify_key"]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} <id={self.node_id.short()}, 🔑={str(self.verify_key)[0:8]}>"

    @classmethod
    def from_client(cls, client: SyftClient) -> Self:
        return cls(node_id=client.id, verify_key=client.credentials.verify_key)


@serializable()
class UserIdentity(Identity):
    """This class is used to identify the data scientist users of the node"""

    pass
