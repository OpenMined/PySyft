"""LocalSignaling carries an offer to the receiver and its answer back."""

import asyncio

import pytest

from syft.sync.connections.webrtc.signaling.local_signaling import LocalSignaling

DO = "do@example.com"
DS = "ds@example.com"
OTHER_DS = "other-ds@example.com"


def ignore_hello(sender: str, session: str) -> None:
    pass


def ignore_error(error: ConnectionError) -> None:
    pass


async def answer_nothing(sender: str, sdp: str) -> str:
    return ""


def test_offer_reaches_listener_and_returns_its_answer():
    async def exchange() -> str:
        signaling = LocalSignaling()

        async def answer(sender: str, sdp: str) -> str:
            return f"answer from {DO} to {sender}: {sdp}"

        await signaling.listen(DO, answer, ignore_hello, ignore_error, ignore_error)
        return await signaling.offer(DS, DO, "offer")

    assert asyncio.run(exchange()) == f"answer from {DO} to {DS}: offer"


def test_offer_to_peer_not_listening_is_refused():
    with pytest.raises(ConnectionError):
        asyncio.run(LocalSignaling().offer(DS, DO, "offer"))


def test_hello_reaches_only_peers_that_listen():
    async def say_hello() -> list[tuple[str, str]]:
        signaling = LocalSignaling()
        hellos: list[tuple[str, str]] = []

        def on_hello(sender: str, session: str) -> None:
            hellos.append((sender, session))

        await signaling.listen(DO, answer_nothing, on_hello, ignore_error, ignore_error)
        await signaling.listen(
            DS, answer_nothing, ignore_hello, ignore_error, ignore_error
        )

        await signaling.hello(DS, [DO, OTHER_DS])
        # LocalSignaling calls on_hello on the loop of the receiver. The call
        # runs when this coroutine yields.
        await asyncio.sleep(0)
        return hellos

    [(sender, session)] = asyncio.run(say_hello())
    assert sender == DS and session
