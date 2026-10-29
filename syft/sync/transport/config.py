"""The connection each transport component uses, as plain config."""

from typing import Self

from pydantic import BaseModel

from syft.sync.connections.base_connection import ConnectionConfig
from syft.sync.transport.protocols import TRANSPORT_COMPONENTS


class TransportConfig(BaseModel):
    """The connection each transport component uses.
    ``SyftboxManager.from_config`` builds the transport from it; only one
    Google Drive connection for every component is built so far."""

    p2p: ConnectionConfig
    owner_store: ConnectionConfig
    watcher_store: ConnectionConfig
    collections: ConnectionConfig
    peering: ConnectionConfig

    @classmethod
    def one_connection(cls, connection_config: ConnectionConfig) -> Self:
        """Every component over the same connection."""
        return cls(**{name: connection_config for name in TRANSPORT_COMPONENTS})

    def connection_configs(self) -> list[ConnectionConfig]:
        return [getattr(self, name) for name in TRANSPORT_COMPONENTS]

    def uses_one_connection(self) -> bool:
        return all(config == self.p2p for config in self.connection_configs())
