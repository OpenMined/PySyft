"""LocalSignaling carries an offer to the receiver and its answer back."""

import asyncio

import pytest

from syft.sync.connections.webrtc.signaling.local_signaling import LocalSignaling

DO = "do@example.com"
DS = "ds@example.com"


def test_offer_reaches_listener_and_returns_its_answer():
    async def exchange() -> str:
        signaling = LocalSignaling()

        async def answer(sender: str, sdp: str) -> str:
            return f"answer from {DO} to {sender}: {sdp}"

        await signaling.listen(DO, answer)
        return await signaling.offer(DS, DO, "offer")

    assert asyncio.run(exchange()) == f"answer from {DO} to {DS}: offer"


def test_offer_to_peer_not_listening_is_refused():
    with pytest.raises(ConnectionError):
        asyncio.run(LocalSignaling().offer(DS, DO, "offer"))
