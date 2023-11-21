# future
from __future__ import annotations

# stdlib
from typing import Callable
from typing import List
from typing import Optional

# third party
from packaging import version
from pydantic import BaseModel
from pydantic import root_validator

# relative
from ...abstract_node import NodeType
from ...node.credentials import SyftVerifyKey
from ...protocol.data_protocol import get_data_protocol
from ...serde.serializable import serializable
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.syft_object import SYFT_OBJECT_VERSION_3
from ...types.syft_object import StorableObjectType
from ...types.syft_object import SyftObject
from ...types.transforms import convert_types
from ...types.transforms import drop
from ...types.transforms import rename
from ...types.transforms import transform
from ...types.uid import UID


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
        raise ValueError(msg)
    if client_syft_version.pre != node_syft_version.pre:
        if not silent:
            print(f"Warning: {msg}")
            return False
    return True


@serializable()
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
    admin_email: Optional[str]


@serializable()
class NodeMetadata(SyftObject):
    __canonical_name__ = "NodeMetadata"
    __version__ = SYFT_OBJECT_VERSION_1

    name: str
    id: UID
    verify_key: SyftVerifyKey
    highest_object_version: int
    lowest_object_version: int
    syft_version: str
    node_type: NodeType = NodeType.DOMAIN
    deployed_on: str = "Date"
    organization: str = "OpenMined"
    on_board: bool = False
    description: str = "Text"
    signup_enabled: bool
    admin_email: str
    node_side_type: str
    show_warnings: bool

    def check_version(self, client_version: str) -> bool:
        return check_version(
            client_version=client_version,
            server_version=self.syft_version,
            server_name=self.name,
        )


@serializable()
class NodeMetadataV2(SyftObject):
    __canonical_name__ = "NodeMetadata"
    __version__ = SYFT_OBJECT_VERSION_2

    name: str
    highest_version: int
    lowest_version: int
    id: UID
    verify_key: SyftVerifyKey
    syft_version: str
    node_type: NodeType = NodeType.DOMAIN
    deployed_on: str = "Date"
    organization: str = "OpenMined"
    on_board: bool = False
    description: str = "Text"
    signup_enabled: bool
    admin_email: str
    node_side_type: str
    show_warnings: bool

    def check_version(self, client_version: str) -> bool:
        return check_version(
            client_version=client_version,
            server_version=self.syft_version,
            server_name=self.name,
        )


@serializable()
class NodeMetadataV3(SyftObject):
    __canonical_name__ = "NodeMetadata"
    __version__ = SYFT_OBJECT_VERSION_3

    name: str
    id: UID
    verify_key: SyftVerifyKey
    highest_version: int
    lowest_version: int
    syft_version: str
    node_type: NodeType = NodeType.DOMAIN
    organization: str = "OpenMined"
    description: str = "Text"
    node_side_type: str
    show_warnings: bool

    def check_version(self, client_version: str) -> bool:
        return check_version(
            client_version=client_version,
            server_version=self.syft_version,
            server_name=self.name,
        )


@serializable()
class NodeMetadataJSON(BaseModel, StorableObjectType):
    metadata_version: int
    name: str
    id: str
    verify_key: str
    highest_object_version: Optional[int]
    lowest_object_version: Optional[int]
    syft_version: str
    node_type: str = NodeType.DOMAIN.value
    organization: str = "OpenMined"
    description: str = "My cool domain"
    signup_enabled: bool = False
    admin_email: str = ""
    node_side_type: str
    show_warnings: bool
    supported_protocols: List = []

    @root_validator(pre=True)
    def add_protocol_versions(cls, values: dict) -> dict:
        if "supported_protocols" not in values:
            data_protocol = get_data_protocol()
            values["supported_protocols"] = data_protocol.supported_protocols
        return values

    def check_version(self, client_version: str) -> bool:
        return check_version(
            client_version=client_version,
            server_version=self.syft_version,
            server_name=self.name,
        )


@transform(NodeMetadataV3, NodeMetadataJSON)
def metadata_to_json() -> List[Callable]:
    return [
        drop(["__canonical_name__"]),
        rename("__version__", "metadata_version"),
        convert_types(["id", "verify_key", "node_type"], str),
        rename("highest_version", "highest_object_version"),
        rename("lowest_version", "lowest_object_version"),
    ]


@transform(NodeMetadataJSON, NodeMetadataV3)
def json_to_metadata() -> List[Callable]:
    return [
        drop(["metadata_version", "supported_protocols"]),
        convert_types(["id", "verify_key"], [UID, SyftVerifyKey]),
        convert_types(["node_type"], NodeType),
        rename("highest_object_version", "highest_version"),
        rename("lowest_object_version", "lowest_version"),
    ]
