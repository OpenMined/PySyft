# stdlib
from enum import Enum
from typing import Any

# relative
from ...client.client import login_as_guest
from ...serde.serializable import serializable
from ...service.metadata.node_metadata import NodeMetadataJSON
from ...service.network.routes import NodeRouteType
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.uid import UID
from ...util.markdown import as_markdown_python_code


@serializable()
class EnclaveStatus(Enum):
    IDLE = "idle"
    NOT_INITIALIZED = "not_initialized"
    INITIALIZING = "initializing"
    BUSY = "busy"
    SHUTTING_DOWN = "shutting_down"


# TODO rename Enclave to EnclaveInstance to prevent a clash
@serializable()
class Enclave(SyftObject):
    # version
    __canonical_name__ = "Enclave"
    __version__ = SYFT_OBJECT_VERSION_1

    node_uid: UID
    name: str
    route: NodeRouteType
    status: EnclaveStatus = EnclaveStatus.NOT_INITIALIZED
    metadata: NodeMetadataJSON | None = None

    __attr_searchable__ = ["name", "route", "status"]
    __repr_attrs__ = ["name", "route", "status", "metadata"]
    __attr_unique__ = ["name"]

    # TODO replace the create method with pydantic field validators, or find a better alternative
    @classmethod
    def create(cls, route: NodeRouteType) -> "Enclave":
        # TODO: find the standard method to convert route to client object
        metadata = login_as_guest(url=route.host_or_ip, port=route.port).metadata
        return cls(
            node_uid=UID(metadata.id),
            name=metadata.name,
            route=route,
            status=cls.get_status(),
            metadata=metadata,
        )

    @classmethod
    def get_status(cls) -> EnclaveStatus:
        # TODO check the actual status of the enclave
        return EnclaveStatus.IDLE

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        return hash(self) == hash(other)

    def __repr_syft_nested__(self) -> str:
        return f"Enclave({self.name})"

    def __repr__(self) -> str:
        return f"<Enclave: {self.name}>"

    def _repr_markdown_(self, wrap_as_python: bool = True, indent: int = 0) -> str:
        _repr_str = f"Enclave: {self.name}\n"
        _repr_str += f"Address: {self.address}\n"
        _repr_str += f"Status: {self.status}\n"
        _repr_str += f"Metadata: {self.metadata}\n"
        return as_markdown_python_code(_repr_str) if wrap_as_python else _repr_str
