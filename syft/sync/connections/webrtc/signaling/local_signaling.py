"""Signaling between peers in one process. TEMP: V3 replaces it with the
peering connection."""

import asyncio
from collections import defaultdict
from dataclasses import dataclass

from syft.sync.connections.webrtc.signaling.signaling import (
    IceServers,
    OnClosed,
    OnOffer,
    OnOnline,
    Signaling,
)


@dataclass(frozen=True)
class Listener:
    on_offer: OnOffer
    on_online: OnOnline
    # The loop that runs both callbacks.
    loop: asyncio.AbstractEventLoop


class LocalSignaling(Signaling):
    """Gives each offer directly to the ``on_offer`` of the receiver.

    ``on_offer`` runs on the loop of the receiver. A peer is online while it
    listens.
    """

    def __init__(self):
        # email -> its listener
        self._listeners: dict[str, Listener] = {}
        # email -> the emails that watch it
        self._watchers: dict[str, set[str]] = defaultdict(set)

    async def listen(
        self, email: str, on_offer: OnOffer, on_online: OnOnline, on_closed: OnClosed
    ) -> None:
        """A session in one process ends only at ``close``, so ``on_closed``
        never runs."""
        self._listeners[email] = Listener(
            on_offer, on_online, asyncio.get_running_loop()
        )
        for watcher in self._watchers[email] & self._listeners.keys():
            self._report_online(watcher, email)

    async def ice_servers(self, email: str) -> IceServers:
        # Peers in one process reach each other with no STUN or TURN server.
        return []

    async def watch(self, email: str, peers: list[str]) -> None:
        for watchers in self._watchers.values():
            watchers.discard(email)
        for peer in peers:
            self._watchers[peer].add(email)
            if peer in self._listeners:
                self._report_online(email, peer)

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

    def _report_online(self, watcher: str, peer: str) -> None:
        listener = self._listeners[watcher]
        listener.loop.call_soon_threadsafe(listener.on_online, peer)
