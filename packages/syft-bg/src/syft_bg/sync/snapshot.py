"""Data contract between sync service and consumers."""

from typing import Optional

from pydantic import BaseModel


class PeerVersionInfo(BaseModel):
    """Version info for compat checks. Slim projection of syft_client VersionInfo."""

    syft_client_version: str
    protocol_version: str


class SyncSnapshot(BaseModel):
    sync_time: float
    sync_count: int = 0
    sync_error: Optional[str] = None
    sync_duration_ms: int = 0

    job_names: list[str] = []
    peer_emails: list[str] = []
    approved_peer_emails: list[str] = []

    own_version: Optional[PeerVersionInfo] = None
    peer_versions: dict[str, PeerVersionInfo] = {}
