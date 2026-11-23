"""LocalSignaling carries an offer to the receiver and its answer back."""

import asyncio

import pytest

from syft.sync.connections.webrtc.signaling.local_signaling import LocalSignaling

DO = "do@example.com"
DS = "ds@example.com"
OTHER_DS = "other-ds@example.com"


def ignore_online(peer: str) -> None:
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

        await signaling.listen(DO, answer, ignore_online, ignore_error, ignore_error)
        return await signaling.offer(DS, DO, "offer")

    assert asyncio.run(exchange()) == f"answer from {DO} to {DS}: offer"


def test_offer_to_peer_not_listening_is_refused():
    with pytest.raises(ConnectionError):
        asyncio.run(LocalSignaling().offer(DS, DO, "offer"))


def test_watch_reports_peers_listening_now_and_later():
    async def watch_two_peers() -> list[str]:
        signaling = LocalSignaling()
        online: list[str] = []
        await signaling.listen(
            DO, answer_nothing, online.append, ignore_error, ignore_error
        )
        await signaling.listen(
            DS, answer_nothing, ignore_online, ignore_error, ignore_error
        )

        await signaling.watch(DO, [DS, OTHER_DS])
        await signaling.listen(
            OTHER_DS, answer_nothing, ignore_online, ignore_error, ignore_error
        )
        # LocalSignaling calls on_online on the loop of the watcher. The calls
        # run when this coroutine yields.
        await asyncio.sleep(0)
        return online

    assert asyncio.run(watch_two_peers()) == [DS, OTHER_DS]


def test_watch_replaces_earlier_watch_list():
    async def watch_then_unwatch() -> list[str]:
        signaling = LocalSignaling()
        online: list[str] = []
        await signaling.listen(
            DO, answer_nothing, online.append, ignore_error, ignore_error
        )

        await signaling.watch(DO, [DS])
        await signaling.watch(DO, [])
        await signaling.listen(
            DS, answer_nothing, ignore_online, ignore_error, ignore_error
        )
        await asyncio.sleep(0)
        return online

    assert asyncio.run(watch_then_unwatch()) == []
