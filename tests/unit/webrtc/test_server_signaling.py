"""ServerSignaling against a signaling server in this process."""

import asyncio

import pytest

from syft.sync.connections.webrtc.signaling.server_signaling import ServerSignaling
from syft.sync.connections.webrtc.webrtc_config import WebRTCConfig
from syft.sync.connections.webrtc.signaling.signaling import OnOnline
from tests.unit.webrtc.test_signaling_server import (
    ACCESS_KEY,
    ICE_SERVERS,
    A,
    B,
    run_with_server,
)


async def echo_answer(sender: str, sdp: str) -> str:
    return f"answer to {sender}: {sdp}"


def ignore_online(peer: str) -> None:
    pass


def ignore_closed(error: ConnectionError) -> None:
    pass


async def listen(
    url: str, email: str, on_online: OnOnline = ignore_online
) -> ServerSignaling:
    signaling = ServerSignaling(WebRTCConfig(signaling_url=url, access_key=ACCESS_KEY))
    await signaling.listen(email, echo_answer, on_online, ignore_closed)
    return signaling


def test_ice_servers_come_from_server():
    async def scenario(url: str) -> None:
        signaling = await listen(url, A)
        assert await signaling.ice_servers(A) == ICE_SERVERS
        await signaling.close(A)

    run_with_server(scenario)


def test_listen_with_wrong_key_raises():
    async def scenario(url: str) -> None:
        signaling = ServerSignaling(WebRTCConfig(signaling_url=url, access_key="wrong"))
        with pytest.raises(ConnectionError, match="The access key is wrong"):
            await signaling.listen(A, echo_answer, ignore_online, ignore_closed)

    run_with_server(scenario)


def test_offer_returns_peer_answer():
    async def scenario(url: str) -> None:
        a, b = await listen(url, A), await listen(url, B)

        assert await a.offer(A, B, "offer") == f"answer to {A}: offer"
        await a.close(A)
        await b.close(B)

    run_with_server(scenario)


def test_offer_to_offline_peer_raises():
    async def scenario(url: str) -> None:
        a = await listen(url, A)

        with pytest.raises(ConnectionError, match=f"{B} is offline"):
            await a.offer(A, B, "offer")
        await a.close(A)

    run_with_server(scenario)


def test_watch_reports_peer_coming_online():
    async def scenario(url: str) -> None:
        reported = asyncio.Queue()
        a = await listen(url, A, reported.put_nowait)
        await a.watch(A, [B])

        b = await listen(url, B)
        assert await reported.get() == B
        await a.close(A)
        await b.close(B)

    run_with_server(scenario)
