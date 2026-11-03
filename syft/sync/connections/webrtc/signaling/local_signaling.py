"""Signaling between peers in one process. TEMP: V3 replaces it with the
peering connection."""

import asyncio

from syft.sync.connections.webrtc.signaling.signaling import OnOffer, Signaling


class LocalSignaling(Signaling):
    """Hands an offer straight to the receiver's ``on_offer``, on its loop."""

    def __init__(self):
        # email -> its on_offer and the loop that runs it
        self._listeners: dict[str, tuple[OnOffer, asyncio.AbstractEventLoop]] = {}

    async def listen(self, email: str, on_offer: OnOffer) -> None:
        self._listeners[email] = (on_offer, asyncio.get_running_loop())

    async def offer(self, sender: str, receiver: str, sdp: str) -> str:
        if receiver not in self._listeners:
            raise ConnectionError(f"{receiver} is not listening for offers")
        on_offer, loop = self._listeners[receiver]
        answer = asyncio.run_coroutine_threadsafe(on_offer(sender, sdp), loop)
        return await asyncio.wrap_future(answer)
