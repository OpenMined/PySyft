"""Signaling between peers in one process."""

import asyncio
import uuid
from dataclasses import dataclass

from syft.sync.connections.webrtc.signaling.signaling import (
    IceServerSet,
    OnAnswerFailed,
    OnClosed,
    OnHello,
    OnOffer,
    Signaling,
)


@dataclass(frozen=True)
class Listener:
    on_offer: OnOffer
    on_hello: OnHello
    # A new id at each listen.
    session: str
    # The loop that runs both callbacks.
    loop: asyncio.AbstractEventLoop


class LocalSignaling(Signaling):
    """Gives each hello and each offer directly to the receiver.

    The callbacks run on the loop of the receiver. A peer is online while it
    listens.
    """

    def __init__(self):
        # email -> its listener
        self._listeners: dict[str, Listener] = {}

    async def listen(
        self,
        email: str,
        on_offer: OnOffer,
        on_hello: OnHello,
        on_closed: OnClosed,
        on_answer_failed: OnAnswerFailed,
    ) -> None:
        """A session in one process ends only at ``close``, so ``on_closed``
        never runs. A failed answer goes to the offerer as an error, so
        ``on_answer_failed`` never runs."""
        self._listeners[email] = Listener(
            on_offer, on_hello, uuid.uuid4().hex, asyncio.get_running_loop()
        )

    async def ice_servers(self, email: str) -> IceServerSet:
        # Peers in one process reach each other with no STUN or TURN server.
        return IceServerSet([], None)

    async def hello(self, email: str, peers: list[str]) -> None:
        session = self._listeners[email].session
        for peer in peers:
            if peer in self._listeners:
                listener = self._listeners[peer]
                listener.loop.call_soon_threadsafe(listener.on_hello, email, session)

    async def offer(self, sender: str, receiver: str, sdp: str) -> str:
        if receiver not in self._listeners:
            raise ConnectionError(f"{receiver} is not listening for offers")
        listener = self._listeners[receiver]
        answer = asyncio.run_coroutine_threadsafe(
            listener.on_offer(sender, sdp), listener.loop
        )
        return await asyncio.wrap_future(answer)

    async def close(self, email: str) -> None:
        del self._listeners[email]
