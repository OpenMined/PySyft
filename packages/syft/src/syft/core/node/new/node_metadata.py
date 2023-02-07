# future
from __future__ import annotations

# stdlib
from typing import Callable
from typing import List

# third party
from packaging import version
from pydantic import BaseModel

# relative
from ....core.node.common.node_table.syft_object import SYFT_OBJECT_VERSION_1
from ....core.node.common.node_table.syft_object import SyftObject
from ....core.node.common.node_table.syft_object import transform
from ...common.serde.serializable import serializable
from ...common.uid import UID
from ..new.credentials import SyftVerifyKey
from ..new.transforms import convert_types
from ..new.transforms import drop
from ..new.transforms import rename


def check_version(
    client_version: str, server_version: str, server_name: str, silent: bool = False
) -> bool:
    client_syft_version = version.parse(client_version)
    node_syft_version = version.parse(server_version)
    msg = (
        f"You are running syft=={client_version} but "
        f"{server_name} node requires {server_version}"
    )
    if client_syft_version.base_version != node_syft_version.base_version:
        raise Exception(msg)
    if client_syft_version.pre != node_syft_version.pre:
        if not silent:
            print(f"Warning: {msg}")
            return False
    return True


@serializable(recursive_serde=True)
class NodeMetadata(SyftObject):
    __canonical_name__ = "NodeMetadata"
    __version__ = SYFT_OBJECT_VERSION_1

    name: str
    id: UID
    verify_key: SyftVerifyKey
    highest_object_version: int
    lowest_object_version: int
    syft_version: str

    def check_version(self, client_version: str) -> None:
        return check_version(
            client_version=client_version,
            server_version=self.syft_version,
            server_name=self.name,
        )


@serializable(recursive_serde=True)
class NodeMetadataJSON(BaseModel):
    metadata_version: int
    name: str
    id: str
    verify_key: str
    highest_object_version: int
    lowest_object_version: int
    syft_version: str

    def check_version(self, client_version: str) -> bool:
        return check_version(
            client_version=client_version,
            server_version=self.syft_version,
            server_name=self.name,
        )


@transform(NodeMetadata, NodeMetadataJSON)
def json_metadata() -> List[Callable]:
    return [
        drop("__canonical_name__"),
        rename("__version__", "metadata_version"),
        convert_types(["id", "verify_key"], str),
    ]
