# future
from __future__ import annotations

# stdlib
from typing import Callable
from typing import List
from typing import Optional

# third party
from packaging import version
from pydantic import BaseModel

# relative
from ..new.credentials import SyftVerifyKey
from ..new.transforms import convert_types
from ..new.transforms import drop
from ..new.transforms import rename
from .serializable import serializable
from .syft_object import SYFT_OBJECT_VERSION_1
from .syft_object import StorableObjectType
from .syft_object import SyftObject
from .transforms import transform
from .uid import UID


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
class NodeMetadataUpdate(SyftObject):
    __canonical_name__ = "NodeMetadataUpdate"
    __version__ = SYFT_OBJECT_VERSION_1

    name: Optional[str]
    organization: Optional[str]
    description: Optional[str]
    on_board: Optional[bool]
    id: Optional[UID]
    verify_key: Optional[SyftVerifyKey]
    highest_object_version: Optional[int]
    lowest_object_version: Optional[int]
    syft_version: Optional[str]


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
    node_type: str = "Domain"
    deployed_on: str = "Date"
    organization: str = "OpenMined"
    on_board: bool = False
    description: str = "Text"

    def check_version(self, client_version: str) -> None:
        return check_version(
            client_version=client_version,
            server_version=self.syft_version,
            server_name=self.name,
        )


@serializable(recursive_serde=True)
class NodeMetadataJSON(BaseModel, StorableObjectType):
    metadata_version: int
    name: str
    id: str
    verify_key: str
    highest_object_version: int
    lowest_object_version: int
    syft_version: str
    node_type: str = "Domain"
    deployed_on: str = "Date"
    organization: str = "OpenMined"
    on_board: bool = False
    description: str = "My cool domain"

    def check_version(self, client_version: str) -> bool:
        return check_version(
            client_version=client_version,
            server_version=self.syft_version,
            server_name=self.name,
        )


@transform(NodeMetadata, NodeMetadataJSON)
def metadata_to_json() -> List[Callable]:
    return [
        drop("__canonical_name__"),
        rename("__version__", "metadata_version"),
        convert_types(["id", "verify_key"], str),
    ]


@transform(NodeMetadataJSON, NodeMetadata)
def json_to_metadata() -> List[Callable]:
    return [
        drop(["metadata_version"]),
        convert_types(["id", "verify_key"], [UID, SyftVerifyKey]),
    ]


class EnclaveMetadata:
    """Contains metadata to connect to a specific Enclave"""

    pass
