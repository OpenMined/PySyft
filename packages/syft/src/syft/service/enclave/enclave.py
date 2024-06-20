# stdlib
from enum import Enum
from typing import Any

# third party
from pydantic import model_validator

# relative
from ...client.client import SyftClient
from ...client.client import login
from ...client.client import login_as_guest
from ...serde.serializable import serializable
from ...service.metadata.node_metadata import NodeMetadataJSON
from ...service.network.routes import NodeRouteType
from ...service.response import SyftError
from ...service.response import SyftException
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


@serializable()
class EnclaveInstance(SyftObject):
    # version
    __canonical_name__ = "EnclaveInstance"
    __version__ = SYFT_OBJECT_VERSION_1

    node_uid: UID
    name: str
    route: NodeRouteType
    status: EnclaveStatus = EnclaveStatus.NOT_INITIALIZED
    metadata: NodeMetadataJSON | None = None

    __attr_searchable__ = ["name", "route", "status"]
    __repr_attrs__ = ["name", "route", "status", "metadata"]
    __attr_unique__ = ["name"]

    @model_validator(mode="before")
    @classmethod
    def initialize_values(cls, values: dict[str, Any]) -> dict[str, Any]:
        if "route" in values:
            route = values["route"]
            metadata = login_as_guest(url=route.host_or_ip, port=route.port).metadata
            if not metadata:
                raise SyftException("Failed to fetch metadata from the node")

            values.update(
                {
                    # 'id': UID(metadata.id),
                    "node_uid": UID(metadata.id),
                    "name": metadata.name,
                    "status": EnclaveStatus.NOT_INITIALIZED,
                    "metadata": metadata,
                }
            )
        return values

    @classmethod
    def get_status(cls) -> EnclaveStatus:
        # TODO check the actual status of the enclave
        return EnclaveStatus.IDLE

    def get_client(self, verify_key: str) -> SyftClient:
        # TODO: find the standard method to convert route to client object
        # TODO for this prototype/demo all communication is done via a DS client.
        # Later, we will use verify keys to authenticate actions of each member in the
        # Enclave. Also, there will be no concept of admin users. This will prevent anyone,
        # including the Enclave owner domain, from performing elevated actions like
        # accessing other member's data.
        PASSWORD = "changethis"  # nosec

        def attempt_login() -> SyftClient | SyftError:
            return login(
                email="ds@openmined.org",
                password=PASSWORD,
                url=self.route.host_or_ip,
                port=self.route.port,
            )

        client = attempt_login()

        if isinstance(client, SyftError):
            login_as_guest(url=self.route.host_or_ip, port=self.route.port).register(
                name="Data Scientist",
                email="ds@openmined.org",
                password=PASSWORD,
                password_verify=PASSWORD,
            )
            client = attempt_login()

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
