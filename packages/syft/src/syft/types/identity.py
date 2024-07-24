# future
from __future__ import annotations

# stdlib
from typing import TYPE_CHECKING

# third party
from typing_extensions import Self

# relative
from ..serde.serializable import serializable
from ..server.credentials import SyftVerifyKey
from .base import SyftBaseModel
from .uid import UID

if TYPE_CHECKING:
    # relative
    from ..client.client import SyftClient


class Identity(SyftBaseModel):
    server_id: UID
    verify_key: SyftVerifyKey

    __repr_attrs__ = ["id", "verify_key"]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} <id={self.server_id.short()}, 🔑={str(self.verify_key)[0:8]}>"

    @classmethod
    def from_client(cls, client: SyftClient) -> Self:
        if not client.credentials:
            raise ValueError(f"{client} has no signing key!")
        return cls(server_id=client.id, verify_key=client.credentials.verify_key)


@serializable(canonical_name="UserIdentity", version=1)
class UserIdentity(Identity):
    """This class is used to identify the data scientist users of the server"""

    pass
