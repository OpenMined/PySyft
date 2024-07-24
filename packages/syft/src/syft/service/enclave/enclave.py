# stdlib
from enum import Enum
from typing import Any

# third party
from pydantic import model_validator

# relative
from ...client.client import SyftClient
from ...client.enclave_client import EnclaveClient
from ...serde.serializable import serializable
from ...server.credentials import SyftSigningKey
from ...service.metadata.server_metadata import ServerMetadataJSON
from ...service.network.routes import ServerRouteType
from ...service.network.server_peer import route_to_connection
from ...service.response import SyftException
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.uid import UID
from ...util.markdown import as_markdown_python_code


@serializable(canonical_name="EnclaveStatus", version=1)
class EnclaveStatus(Enum):
    IDLE = "idle"
    NOT_INITIALIZED = "not_initialized"
    INITIALIZING = "initializing"
    BUSY = "busy"
    SHUTTING_DOWN = "shutting_down"


@serializable()
class EnclaveInstance(SyftObject):
    # version
    __canonical_name__ = "EnclaveInstance"
    __version__ = SYFT_OBJECT_VERSION_1

    server_uid: UID
    name: str
    route: ServerRouteType
    status: EnclaveStatus = EnclaveStatus.NOT_INITIALIZED
    metadata: ServerMetadataJSON | None = None

    __attr_searchable__ = ["name", "route", "status"]
    __repr_attrs__ = ["name", "route", "status", "metadata"]
    __attr_unique__ = ["name"]

    @model_validator(mode="before")
    @classmethod
    def initialize_values(cls, values: dict[str, Any]) -> dict[str, Any]:
        is_being_created = "id" not in values

        if is_being_created and "route" in values:
            connection = route_to_connection(values["route"])
            metadata = connection.get_server_metadata(credentials=None)
            if not metadata:
                raise SyftException("Failed to fetch metadata from the server")

            values.update(
                {
                    "server_uid": UID(metadata.id),
                    "name": metadata.name,
                    "status": cls.get_status(),
                    "metadata": metadata,
                }
            )
        return values

    @classmethod
    def get_status(cls) -> EnclaveStatus:
        # TODO check the actual status of the enclave
        return EnclaveStatus.IDLE

    def get_client(self, credentials: SyftSigningKey) -> SyftClient:
        connection = route_to_connection(route=self.route)
        client = EnclaveClient(connection=connection, credentials=credentials)
        return client

    def get_guest_client(self) -> SyftClient:
        connection = route_to_connection(route=self.route)
        client = EnclaveClient(
            connection=connection, credentials=SyftSigningKey.generate()
        )
        return client

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
        _repr_str += f"Route: {self.route}\n"
        _repr_str += f"Status: {self.status}\n"
        _repr_str += f"Metadata: {self.metadata}\n"
        return as_markdown_python_code(_repr_str) if wrap_as_python else _repr_str
