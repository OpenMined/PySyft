# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import KeysView
from typing import Union

# third party
from nacl.encoding import HexEncoder
from nacl.signing import VerifyKey
from pydantic import BaseModel

# relative
from ......core.node.common.exceptions import InvalidNodeCredentials


class NodeCredentials(BaseModel):
    node_uid: str
    node_name: str
    node_type: str
    verify_key: str

    # allows splatting with **payload
    def keys(self) -> KeysView[str]:
        return self.__dict__.keys()

    # allows splatting with **payload
    def __getitem__(self, key: str) -> Any:
        return self.__dict__.__getitem__(key)

    @staticmethod
    def from_objs(*args: Any, **kwargs: Any) -> NodeCredentials:
        # TODO: we should investigate a way to automatically coerce the right types
        # back and forth with Pydantic and our storage layer
        return NodeCredentials(
            node_uid=kwargs["node_uid"].no_dash,  # type: ignore
            node_name=kwargs["node_name"],
            node_type=kwargs["node_type"],
            verify_key=kwargs["verify_key"].encode(encoder=HexEncoder).decode("utf-8"),  # type: ignore
        )

    def validate(self, key: Union[str, VerifyKey]) -> bool:
        if isinstance(key, VerifyKey):
            verify_key = VerifyKey(self.verify_key, encoder=HexEncoder)
        else:
            verify_key = self.verify_key

        if verify_key == key:
            return True

        raise InvalidNodeCredentials(
            f"Credentials for {self.node_name}: {self.node_uid} do not match"
        )
