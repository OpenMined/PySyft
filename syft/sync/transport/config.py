"""Which implementation each transport component uses, as plain config."""

from pydantic import BaseModel

DRIVE = "drive"


class TransportConfig(BaseModel):
    """Which implementation each transport component uses.
    ``SyftboxManager.from_config`` builds the transport from it; only Google
    Drive for every component is built so far."""

    p2p: str = DRIVE
    owner_store: str = DRIVE
    watcher_store: str = DRIVE
    collections: str = DRIVE
    peering: str = DRIVE

    def all_drive(self) -> bool:
        return set(self.model_dump().values()) == {DRIVE}
