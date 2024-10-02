# stdlib
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING

# relative
from .serde.serializable import serializable
from .store.db.db import DBConfig
from .store.db.db import DBManager
from .types.uid import UID

if TYPE_CHECKING:
    # relative
    from .server.service_registry import ServiceRegistry
    from .service.service import AbstractService


@serializable(canonical_name="ServerType", version=1)
class ServerType(str, Enum):
    DATASITE = "datasite"
    NETWORK = "network"
    ENCLAVE = "enclave"
    GATEWAY = "gateway"

    def __str__(self) -> str:
        # Use values when transforming ServerType to str
        return self.value


@serializable(canonical_name="ServerSideType", version=1)
class ServerSideType(str, Enum):
    LOW_SIDE = "low"
    HIGH_SIDE = "high"

    def __str__(self) -> str:
        return self.value


class AbstractServer:
    id: UID | None
    name: str | None
    server_type: ServerType | None
    server_side_type: ServerSideType | None
    in_memory_workers: bool
    services: "ServiceRegistry"
    db_config: DBConfig
    db: DBManager[DBConfig]

    def get_service(self, path_or_func: str | Callable) -> "AbstractService":
        raise NotImplementedError
