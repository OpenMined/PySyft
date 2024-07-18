# future
from __future__ import annotations

# stdlib
from collections.abc import Callable

# third party
from packaging import version
from pydantic import BaseModel
from pydantic import model_validator

# relative
from ...abstract_server import ServerType
from ...protocol.data_protocol import get_data_protocol
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...types.syft_object import SYFT_OBJECT_VERSION_1
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
    server_syft_version = version.parse(server_version)
    msg = (
        f"You are running syft=={client_version} but "
        f"{server_name} server requires {server_version}"
    )
    if client_syft_version.base_version != server_syft_version.base_version:
        raise ValueError(msg)
    if client_syft_version.pre != server_syft_version.pre:
        if not silent:
            print(f"Warning: {msg}")
            return False
    return True


@serializable()
class ServerMetadata(SyftObject):
    __canonical_name__ = "ServerMetadata"
    __version__ = SYFT_OBJECT_VERSION_1

    name: str
    id: UID
    verify_key: SyftVerifyKey
    highest_version: int
    lowest_version: int
    syft_version: str
    server_type: ServerType = ServerType.DATASITE
    organization: str = "OpenMined"
    description: str = "Text"
    server_side_type: str
    show_warnings: bool
    eager_execution_enabled: bool
    min_size_blob_storage_mb: int

    def check_version(self, client_version: str) -> bool:
        return check_version(
            client_version=client_version,
            server_version=self.syft_version,
            server_name=self.name,
        )


@serializable(canonical_name="ServerMetadataJSON", version=1)
class ServerMetadataJSON(BaseModel, StorableObjectType):
    metadata_version: int
    name: str
    id: str
    verify_key: str
    highest_object_version: int | None = None
    lowest_object_version: int | None = None
    syft_version: str
    server_type: str = ServerType.DATASITE.value
    organization: str = "OpenMined"
    description: str = "My cool datasite"
    signup_enabled: bool = False
    eager_execution_enabled: bool = False
    admin_email: str = ""
    server_side_type: str
    show_warnings: bool
    supported_protocols: list = []
    min_size_blob_storage_mb: int

    @model_validator(mode="before")
    @classmethod
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


@transform(ServerMetadata, ServerMetadataJSON)
def metadata_to_json() -> list[Callable]:
    return [
        drop(["__canonical_name__"]),
        rename("__version__", "metadata_version"),
        convert_types(["id", "verify_key", "server_type"], str),
        rename("highest_version", "highest_object_version"),
        rename("lowest_version", "lowest_object_version"),
    ]


@transform(ServerMetadataJSON, ServerMetadata)
def json_to_metadata() -> list[Callable]:
    return [
        drop(["metadata_version", "supported_protocols"]),
        convert_types(["id", "verify_key"], [UID, SyftVerifyKey]),
        convert_types(["server_type"], ServerType),
        rename("highest_object_version", "highest_version"),
        rename("lowest_object_version", "lowest_version"),
    ]
