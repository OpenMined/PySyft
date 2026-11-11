"""Helpers for managers whose p2p connection runs over real WebRTC."""

from syft.sync.connections.webrtc.webrtc_p2p import WebRTCP2PConnection
from syft.sync.syftbox_manager import SyftboxManager


def p2p_of(manager: SyftboxManager) -> WebRTCP2PConnection:
    return manager._connection_router.connections.p2p_con


def flush(*managers: SyftboxManager) -> None:
    """Wait until the peers stored every message these managers sent."""
    for manager in managers:
        p2p_of(manager).wait_until_delivered()


def close(*managers: SyftboxManager) -> None:
    for manager in managers:
        p2p_of(manager).close()
